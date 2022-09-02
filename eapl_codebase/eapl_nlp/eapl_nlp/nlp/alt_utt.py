import requests
import json
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import random
import time
import tensorflow.compat.v1 as tf
from multiprocessing import Pool
import logging
from eapl.df.eapl_kpi import eapl_convert_to_list

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs
alt_utt_logger = logging.getLogger(__name__)

gt_lang_list = ['am', 'ar', 'eu', 'bn', 'en-GB', 'pt-BR', 'bg', 'ca', 'chr', 'hr', 'cs', 'da', 'nl', 'et', 'fil',
                'fi', 'fr', 'de', 'el', 'gu', 'iw', 'hi', 'hu', 'is', 'id', 'it', 'ja', 'kn', 'ko', 'lv', 'lt', 'ms',
                'ml', 'mr', 'no', 'pl', 'pt-PT', 'ro', 'ru', 'sr', 'zh-CN', 'sk', 'sl', 'es', 'sw',
                'sv', 'ta', 'te', 'th', 'zh-TW', 'tr', 'ur', 'uk', 'vi', 'cy']


def google_translate(df, col_name, max_utt):
    all_df = pd.DataFrame()
    for utt_count in range(1):
        tmp_df = df.copy()
        sent = tmp_df[col_name].to_list()
        l1, l2 = random.choice(gt_lang_list), random.choice(gt_lang_list)
        lang = ['en', l1, l2, 'en']
        tr_sent = sent
        for i in range(len(lang) - 1):
            src_l = lang[i]
            dest_l = lang[i + 1]
            if src_l == dest_l:
                continue
            try:
                r = requests.post(
                    'https://translation.googleapis.com/language/translate/v2',
                    data={
                        'q': tr_sent,
                        'source': src_l,
                        'target': dest_l,
                        'format': 'text',
                        'model': 'nmt',
                        'key': 'AIzaSyBYMD7n9im4It91F5CT5BeLrfESKS6IXYU'
                    },
                )
            except Exception as e:
                alt_utt_logger.debug("Warning: Exception with Translate API")
                tr_sent = None
                time.sleep(2)
                break

            rd = json.loads(r.text)
            if rd.get("data", None):
                tr_sent = pd.DataFrame(rd['data']['translations'])['translatedText'].to_list()
                tmp_df['alt_utts'] = tr_sent
            else:
                alt_utt_logger.debug("Warning: Incompatible language translation request")
                tr_sent = None
                break

            if tr_sent and dest_l == 'en':
                tmp_df['Lang'] = str(lang)
                all_df = pd.concat([all_df, tmp_df])

    return all_df


def create_alt_utterance_gt(df, col_name, max_utt=10, thread=20, graph=None, embed_fn=None):
    all_df = pd.DataFrame()
    alt_utt_logger.debug(f'Translate request  start')
    process = max_utt * 4

    p = Pool(thread)

    output = p.starmap(google_translate, [[df, col_name, max_utt]] * process)

    for i in output:
        all_df = pd.concat([all_df, i], axis=0)

    p.terminate()

    alt_utt_logger.debug(f'Translate request End')

    final_df = pd.DataFrame()

    alt_utt_logger.debug(f'Ranking of sent start')
    with tf.Session(graph=graph) as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        for id in list(all_df.id.unique()):
            sub_df = all_df[all_df.id == id]
            sent = sub_df.utt.unique()
            utt_list = sub_df['alt_utts'].to_list()
            alt_utt_logger.debug(f'Each input alt utterance rank start: {id}')
            utt_list = rank_alt_utt(sent, utt_list, session, embed_fn)
            alt_utt_logger.debug(f'Each input alt utterance  rank End: {id}')
            topn_utt = utt_list[:max_utt]
            sub_df = sub_df.head(len(topn_utt)).drop(columns=['alt_utts'])
            sub_df['alt_utts'] = topn_utt
            final_df = pd.concat([final_df, sub_df])

    alt_utt_logger.debug(f'Ranking of sent End')

    return final_df


def rank_utt_corr(embedding, match_utt_l):
    emb_1 = embedding[0].reshape((1, -1))
    corr = np.inner(emb_1, embedding)

    df = pd.DataFrame.from_dict({'utt': match_utt_l, 'corr': corr[0]})
    df = df.sort_values(by='corr', ascending=False).reset_index(drop=True)
    df['utt_lc'] = df['utt'].str.lower()
    df = df.drop_duplicates(subset='utt_lc')
    ranked_utt = list(df['utt'])

    return ranked_utt


def rank_alt_utt(sent, utt_list, session, embed_fn):
    utt_list = [i.strip() for i in utt_list]
    sent = [i.strip() for i in sent]
    match_utt_l = list(set(utt_list) - set(sent))
    match_utt_l.append(sent[0])

    embedding = session.run(embed_fn(match_utt_l))
    ranked_utt = rank_utt_corr(embedding, match_utt_l)
    return ranked_utt


def rank_related_utt(alt_utt_df, graph=None, embed_fn=None):
    df = alt_utt_df.copy().reset_index(drop=True)
    df['orank'] = df['Node Intent'].astype('category').cat.codes
    df['arank'] = df.index
    df = df.sort_values('orank')
    org_utt_df = df.drop_duplicates('orank')
    org_utt = org_utt_df['Node Intent'].tolist()
    alt_utt = df['alt_utts'].tolist()

    with tf.Session(graph=graph) as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        emb_org = session.run(embed_fn(org_utt))
        emb_alt = session.run(embed_fn(alt_utt))
    corr_oa = np.inner(emb_alt, emb_org)
    corr_oo = np.inner(emb_org, emb_org)
    df_corr_oa = pd.DataFrame(corr_oa)
    cols = df_corr_oa.columns.tolist()
    cols = [f"v{c}" for c in cols]
    df_corr_oa.columns = cols
    df_corr_oa['arank'] = np.where(True, df['arank'], np.NaN)
    df_corr_oa = df_corr_oa.merge(df, how='left', on='arank')
    df_corr_oo = pd.DataFrame(corr_oo)
    df_corr_oo.columns = cols
    df_corr_oo['orank'] = np.where(True, org_utt_df['orank'], np.NaN)
    df_corr_oa = df_corr_oa.merge(df_corr_oo, how='left', on='orank')
    expr_list = [f"({c}_x * {c}_y)" for c in cols]
    eval_expr = " + ".join(expr_list)
    df_corr_oa['match_score'] = df_corr_oa.eval(eval_expr)
    df_corr_oa = df_corr_oa.sort_values(by=['orank', 'match_score'], ascending=[True, False])

    return df_corr_oa[['alt_utts', 'Node#', 'Node Intent']]


def convert_df_to_luis_json(df, bot_name='c2c_luis'):
    luis_dict = df.to_dict('records')
    return luis_dict


def eapl_alt_utt_init(data, cfg):
    alt_utt_logger.debug(f'Model loading start')
    api_method = cfg.get('api_method', 'gt_api')
    module_url = cfg.get('module_url', 'https://tfhub.dev/google/universal-sentence-encoder/2')
    au_graph_key = cfg.get('au_graph_key', 'au_graph_key')
    au_graph = nlp_glob.get(au_graph_key, None)
    au_embed_key = cfg.get('au_embed_key', 'au_embed_key')
    au_embed = nlp_glob.get(au_embed_key, None)

    # Import the Universal Sentence Encoder's TF Hub module
    if api_method == 'gt_api':
        if au_embed is None:
            if au_graph is None:
                au_graph = tf.Graph()
            with tf.Session(graph=au_graph):
                au_embed = hub.Module(module_url)

            nlp_glob[au_graph_key] = au_graph
            nlp_glob[au_embed_key] = au_embed

        data[au_graph_key] = au_graph
        data[au_embed_key] = au_embed

    alt_utt_logger.debug(f'Model loading End')
    return data


def eapl_alt_utt_rec(data, text_dict, op_dict):
    au_graph_key = op_dict.get('au_graph_key', 'au_graph_key')
    au_graph = data[au_graph_key]
    au_embed_key = op_dict.get('au_embed_key', 'au_embed_key')
    au_embed = data[au_embed_key]
    max_utt = op_dict.get('max_utt', 10)
    input_key = op_dict.get('input_key')
    output_key = op_dict.get('output_key', input_key)
    list_index = op_dict.get('list_index', 0)
    txt_lst = eapl_convert_to_list(text_dict[input_key])

    text_dict[output_key] = []
    if len(txt_lst) > 0:
        txt = txt_lst[list_index]
        df = pd.DataFrame({'id': [1], 'utt': [txt]})
        sent_df = create_alt_utterance_gt(df, max_utt=max_utt, col_name='utt', graph=au_graph, embed_fn=au_embed)
        alt_utts = list(sent_df['alt_utts'].unique())
        text_dict[output_key] = alt_utts
    return text_dict


alt_utt_func_map = {
    'eapl_alt_utt_init': eapl_alt_utt_init,
    'eapl_alt_utt_rec': eapl_alt_utt_rec,
}
nlp_func_map.update(alt_utt_func_map)


def test_alt_utt():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['eapl_alt_utt_init', 'record_nlp_ops'],

        'eapl_alt_utt_init': {
            'func': 'eapl_alt_utt_init'
        },

        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    "op": "eapl_alt_utt_rec",
                    "input_key": 'txt',
                    "output_key": "questions",
                    'max_utt': 4
                },
            ]
        },
    }

    data = {
        'text_obj': [
            {
                "txt": "How to assign alternate managers in SuccessFactors Learning?"
            },
            {
                "txt": "How to record costs for learning events in SuccessFactors Learning?"
            },
        ]
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['text_obj'])


if __name__ == '__main__':
    test_alt_utt()
