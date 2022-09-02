import json

import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer, util
import logging
from sklearn.metrics.pairwise import paired_cosine_distances as paired_distances
import pandas as pd

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs

nlp_p_logger = logging.getLogger(__name__)


def eapl_use_init(data, cfg):
    model_url = cfg.get("model_url", 'https://tfhub.dev/google/universal-sentence-encoder/4')
    model_key = cfg.get('model_key', 'use_model')
    embed = nlp_glob.get(model_key, None)

    if embed is None:
        nlp_p_logger.debug(f"Downloading model from url {model_url}")
        embed = hub.load(model_url)
        nlp_glob[model_key] = embed
        nlp_p_logger.debug(f"Downloading Completed")

    data[model_key] = embed
    return data


# Generate embeddings for list of sentences
def eapl_sent2vec_batch(data, text_dict, op_dict):
    model_key = op_dict['model_key']
    input_key = op_dict['input_key']
    out_key = op_dict.get('out_key', 'embeddings')
    embed = data[model_key]

    sent_list = text_dict[input_key]
    embeddings = embed(sent_list)

    text_dict[out_key] = embeddings
    return text_dict


def eapl_sent2vec(data, text_dict, op_dict):
    input_key = op_dict['input_key']
    out_key = op_dict.get('out_key', 'embedding')

    op_dict['out_key'] = out_key

    text_dict_tmp = {
        input_key: [text_dict[input_key]]
    }
    text_dict_tmp = eapl_sent2vec_batch(data, text_dict_tmp, op_dict)

    text_dict[out_key] = text_dict_tmp[out_key][0]
    return text_dict


def smntc_drop_dplct(start_index, df, diverse):
    tmp_vec = df.iloc[[(start_index - 1)]]['vector'].to_list()[0]
    tmp_df = df.iloc[start_index:]
    tmp_df2 = df.iloc[:start_index]
    index = [i for i in range(len(tmp_vec))]
    tmp_df['distance_tmp'] = (1 - paired_distances([tmp_vec] * tmp_df.shape[0], tmp_df[index]))
    tmp_df = tmp_df[tmp_df['distance_tmp'] < diverse]
    tmp_df2 = pd.concat([tmp_df2, tmp_df])
    return tmp_df2


def eapl_semantic_drop_duplicate(data, text_dict, op_dict):
    text_key = op_dict["text_key"]
    diverse = op_dict.get('diverse', 0.95)
    out_key = op_dict.get('out_key', text_key)
    text_data_type = op_dict.get('text_data_type', 'list')
    dct_text_field = op_dict.get('dct_text_field', None)  # Required only if text_data_type == 'list_of_dicts'
    model_key = op_dict['model_key']

    if text_data_type == 'list':
        embed = data[model_key]
        sent_list = text_dict[text_key]
        embeddings = embed(sent_list)

        txt_lst = text_dict[text_key]
        tmp_df = pd.DataFrame(
            {'tmp_txt': txt_lst, 'vector': embeddings, 'distance_tmp': [0] * len(txt_lst)})
    elif text_data_type == 'list_of_dicts':
        input_key = op_dict['input_key']
        embed = data[model_key]
        txt_ldct = text_dict[text_key]
        tmp_df = pd.DataFrame(txt_ldct)
        tmp_df_cols = list(tmp_df.columns)
        sent_list = list(tmp_df[input_key])
        vectors = embed(sent_list)
        tmp_df['vector'] = vectors
        tmp_df['distance_tmp'] = [0] * len(txt_ldct)
        tmp_df = tmp_df.rename(columns={dct_text_field: 'tmp_txt'})

    tmp_df = pd.concat([tmp_df, tmp_df.vector.apply(pd.Series)], axis=1)

    for i in range(1, len(tmp_df)):
        if i >= len(tmp_df):
            break
        tmp_df = smntc_drop_dplct(start_index=i, df=tmp_df, diverse=diverse)

    if text_data_type == 'list':
        text_dict[out_key] = tmp_df['tmp_txt'].to_list()
    elif text_data_type == 'list_of_dicts':
        tmp_df = tmp_df.rename(columns={'tmp_txt': dct_text_field})[tmp_df_cols]
        text_dict[out_key] = tmp_df.to_dict(orient='records')

    return text_dict


def eapl_semantic_cond_match(data, text_dict, op_dict):
    x_params = op_dict["x_params"]
    y_params = op_dict["y_params"]
    out_key = op_dict.get('out_key', x_params['input_key'])
    filter_cond = op_dict.get("filter_cond", "dist < 0.9")  # defaults to drop_duplicate for similar items
    sort_params = op_dict.get("sort_params", None)
    model_key = op_dict['model_key']
    embed = data[model_key]

    df_dict = {}
    dct_list = [x_params, y_params]
    for idx, dct in enumerate(dct_list):
        input_key = dct["input_key"]
        txt_key = dct.get("txt_key", "txt")
        vec_key = dct.get("vec_key", None)
        td_flag = dct.get("td_flag", True)
        dtype = dct.get("dtype", "dict")  # Options "dict", "list"
        if td_flag:
            dct_input = text_dict[input_key]
        else:
            dct_input = data[input_key]
        if dtype == "dict":
            dct_df = pd.DataFrame(dct_input)
        else:
            dct_df = pd.DataFrame({txt_key: dct_input})

        df_dict[f"orig_{idx}"] = dct_df.copy()
        if not vec_key:
            vec_key = "vector"
            sent_list = list(dct_df[txt_key])
            dct_df[vec_key] = embed(sent_list)
        dct_df.rename(columns={txt_key: "txt", vec_key: "vector"}, inplace=True)
        dct_df['mxy'] = 1
        dct_df = dct_df[["txt", "vector", "mxy"]]
        df_dict[idx] = dct_df

    x_df = df_dict[0]
    y_df = df_dict[1]

    xy_df = x_df.merge(y_df, on='mxy', how='outer')
    x_vec = xy_df['vector_x'].to_list()
    y_vec = xy_df['vector_y'].to_list()

    xy_df['dist'] = 1 - paired_distances(x_vec, y_vec)
    xy_df = xy_df.sort_values(by=['txt_x', 'dist'], ascending=[True, False]).drop_duplicates(subset="txt_x")
    xy_df = xy_df.query(filter_cond)[['txt_x', 'dist']]

    txt_key = x_params.get("txt_key", "txt")
    x_org_df = df_dict["orig_0"]
    x_org_df_cols = x_org_df.columns
    x_org_df = x_org_df.merge(xy_df, left_on=txt_key, right_on='txt_x', how='inner')
    if sort_params:
        x_org_df = x_org_df.sort_values(**sort_params)

    x_dict = x_org_df[x_org_df_cols].to_dict(orient='records')
    text_dict[out_key] = x_dict
    return text_dict


def eapl_use_init_sentence_transformers(data, cfg):
    model_name = cfg.get("model_name", 'distilbert-base-nli-stsb-mean-tokens')
    model_key = cfg.get('model_key', 'use_model_sentencetransformers')
    sentence_transformers_model = nlp_glob.get(model_key, None)

    if sentence_transformers_model is None:
        nlp_p_logger.debug(f"Downloading sentence transformers model {model_name}")
        sentence_transformers_model = SentenceTransformer(model_name)
        nlp_glob[model_key] = sentence_transformers_model
        nlp_p_logger.debug(f"Downloading Completed")

    data[model_key] = sentence_transformers_model
    return data


def eapl_semantic_drop_duplicate_st(data, text_dict, op_dict):
    input_key = op_dict["input_key"]
    text_key = op_dict["text_key"]
    diverse = op_dict.get('diverse', 0.95)
    out_key = op_dict.get('out_key', input_key)
    model_key = op_dict.get('model_key', "use_model_sentencetransformers")
    model_params = op_dict.get('model_params', {})
    unq_val_key = op_dict.get('unq_val_key', 'mval')
    model = data[model_key]
    sentences_df = pd.DataFrame(text_dict[input_key]).drop_duplicates(subset=unq_val_key)

    try:
        sentences = sentences_df[text_key].values.tolist()
        paraphrases = util.paraphrase_mining(model, sentences, **model_params)
        drop_sentence_list = []

        for paraphrase in paraphrases:
            score, i, j = paraphrase
            if score > diverse:
                drop_sentence_list += [sentences[j]]

        clean_sentences_df = sentences_df[~sentences_df[text_key].isin(drop_sentence_list)]
    except:
        clean_sentences_df = sentences_df
    text_dict[out_key] = clean_sentences_df.to_dict(orient='records')
    return text_dict


use_embed_map = {
    'eapl_use_init': eapl_use_init,
    'eapl_use_sent2vec_batch': eapl_sent2vec_batch,
    'eapl_use_sent2vec': eapl_sent2vec,
    'eapl_semantic_drop_duplicate': eapl_semantic_drop_duplicate,
    'eapl_semantic_cond_match': eapl_semantic_cond_match,
    'eapl_semantic_drop_duplicate_st': eapl_semantic_drop_duplicate_st,
    'eapl_use_init_sentence_transformers': eapl_use_init_sentence_transformers
}
nlp_func_map.update(use_embed_map)


def test_use_pipeline():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['eapl_use_init_sentence_transformers', 'record_use_embed'],
        'init_use': {
            'func': 'eapl_use_init',
            'model_url': 'https://tfhub.dev/google/universal-sentence-encoder/4',
            'model_key': 'use_model'
        },
        'eapl_use_init_sentence_transformers': {
            'func': 'eapl_use_init_sentence_transformers',
            'model_name': 'distilbert-base-nli-stsb-mean-tokens',
            'model_key': 'use_model_sentencetransformers'
        },
        'record_use_embed': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    'op': 'eapl_semantic_drop_duplicate_st',
                    'model_key': 'use_model_sentencetransformers',
                    "input_key": "tags",
                    "text_key": "mval",
                    "diverse": 0.7,
                    "out_key": "tags_semantic_drop"
                }
            ]
        }
    }

    data_test = json.loads(
        requests.get("https://s3.amazonaws.com/emplay.botresources/test_data_files/simpplr_test_data_205.json").text)
    func = nlp_func_map['eapl_data_process_fk_flow']
    data_test = func(data_test, nlp_cfg)
    pprint(data_test['text_obj'])


if __name__ == '__main__':
    test_use_pipeline()
