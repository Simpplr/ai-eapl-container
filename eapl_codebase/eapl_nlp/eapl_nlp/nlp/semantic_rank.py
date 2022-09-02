import fasttext
import re
import pandas as pd
from sklearn.metrics.pairwise import paired_cosine_distances as paired_distances
import numpy as np
import logging

try:
    from .nlp_glob import nlp_func_map
except ImportError:
    from nlp_glob import nlp_func_map

semantic_rank_logger = logging.getLogger(__name__)


def remove_stopwords(nlp, string, token=True):
    string = str(string).lower()
    word_tokens = nlp(string)
    filtered_sentence = [w for w in word_tokens if not w.is_stop]
    if token:
        return filtered_sentence
    else:
        return " ".join(filtered_sentence)


def remove_special_character(string, wt=" "):
    filtered_sentence = string
    if type(string) != 'str':
        filtered_sentence = ' '.join(str(v) for v in filtered_sentence)

    tmp_string = re.sub(r"[^a-zA-Z0-9_ ]+", wt, filtered_sentence)

    return tmp_string


def sent2vec(string, nlp, model):
    string = remove_stopwords(nlp, string)
    string = remove_special_character(string)
    vector = model.get_sentence_vector(string)
    return vector


def wgtd_semantic(row, vec, model, nlp, rank_thr_lst=None):
    input_vec = np.asarray(sent2vec(row['input'], nlp=nlp, model=model))
    distance = 0
    n = len(vec)
    d2 = 0
    if rank_thr_lst is None:
        rank_thr_lst = [i for i in range(1, 10)]

    wsum = sum(rank_thr_lst[:n])
    for i in range(n):
        sim_distance = (paired_distances([input_vec], np.asarray([vec[i]])))[0]
        d2 = d2 + (1 - sim_distance)
        wsd = ((rank_thr_lst[i]) / wsum) * (1 - sim_distance)
        distance = distance + wsd

    return distance


def smntc_drop_dplct(start_index, df, diverse):
    tmp_vec = df.iloc[[(start_index - 1)]]['vector'].to_list()[0]
    tmp_df = df.loc[start_index:]
    tmp_df2 = df.loc[:(start_index - 1)]
    index = [i for i in range(len(tmp_vec))]
    tmp_df['distance_tmp'] = (1 - paired_distances([tmp_vec] * tmp_df.shape[0], tmp_df[index]))
    tmp_df = tmp_df[tmp_df['distance_tmp'] < diverse]
    tmp_df2 = pd.concat([tmp_df2, tmp_df])
    return tmp_df2


def qg_check_ents(ent_lst, txt):
    ent_txt = " ".join(ent_lst)
    for ent in ent_txt.split():
        if ent.lower() not in txt.lower():
            return False

    return True


def qg_tokens_by_case(txt, exc_lst=None):
    txt_lst = txt.split()
    if exc_lst is None:
        exc_list = ['how', 'what', 'when', 'where', 'why', 'who', 'whom', 'which', 'whose', 'configure', 'setup',
                    'install', 'guide']
    tu_toks = [t for t in txt_lst[1:] if (t[0].isupper() and (t.lower() not in exc_list))]
    return list(set(tu_toks))


def qg_tok_based_cleanup(ques_lst, pinput):
    pinput_s = re.sub(r'[^\w\s]', '', " ".join(pinput))
    cln_ques_toks = [qg_tokens_by_case(re.sub(r'[^\w\s]', '', q)) for q in ques_lst]
    tok_flg_lst = [qg_check_ents(toks_lst, pinput_s) for toks_lst in cln_ques_toks]
    filt_qs = [q for i, q in enumerate(ques_lst) if tok_flg_lst[i]]
    return filt_qs


def word_difrnc_cnt(string):
    b_lst = string.split(' ')
    return (len(b_lst) - len(set(b_lst))) / len(b_lst)


def semantic_sim_model_init(data, cfg):
    model_path = cfg['model_path']
    model_key = cfg.get('model_key', 'ft_model')
    model = fasttext.load_model(model_path)
    data[model_key] = model
    return data


def semantic_sim_record(data, text_dict, op_dict):
    # Mapping keys and extracting values
    ref_key = op_dict.get('ref_key')
    input_key = op_dict.get('input_key')
    output_key = op_dict.get('output_key', input_key)
    cut_off = op_dict.get('cut_off', 0.80)
    diverse = op_dict.get('diversity_thr', 0.95)
    wrd_cnt_cuttoff = op_dict.get('wrd_cnt_cuttoff', 0.5)
    nlp_key = op_dict.get('nlp_key', 'nlp')
    model_key = op_dict['model_key']
    rank_thr_lst = op_dict.get('rank_thr_lst', None)

    input_lst = text_dict[input_key]
    ref_lst = text_dict[ref_key].copy()
    model = data[model_key]
    nlp = data[nlp_key]

    # Data Processing
    ranked_lst = []
    if ref_lst and input_lst:
        ref_lst.append(" ".join(ref_lst))
        ref_vec = [sent2vec(sent, nlp, model) for sent in ref_lst]
        input_lst_lwr = [i.lower() for i in input_lst]
        tmp_df = pd.DataFrame(
            {'ref': [ref_lst] * len(input_lst), 'input': input_lst, 'input_lwr': input_lst_lwr})
        tmp_df['distance'] = tmp_df.apply(wgtd_semantic, axis=1, vec=ref_vec, model=model, nlp=nlp,
                                          rank_thr_lst=rank_thr_lst)
        tmp_df = tmp_df.query(f"distance > {cut_off}")
        tmp_df['wrd_cnt_rto'] = tmp_df['input_lwr'].apply(word_difrnc_cnt)
        tmp_df = tmp_df.query(f"wrd_cnt_rto < {wrd_cnt_cuttoff}")
        tmp_df = tmp_df.sort_values(by='distance', ascending=False).reset_index(drop=True)
        tmp_df['vector'] = tmp_df['input'].apply(sent2vec, nlp=nlp, model=model)
        tmp_df = pd.concat([tmp_df, tmp_df.vector.apply(pd.Series)], axis=1)
        tmp_df['distance_tmp'] = 0
        for i in range(1, len(tmp_df)):
            if i >= len(tmp_df):
                break
            tmp_df = smntc_drop_dplct(start_index=i, df=tmp_df, diverse=diverse)

        ranked_lst = tmp_df['input'].to_list()

        if len(ranked_lst) > 0:
            ranked_lst = qg_tok_based_cleanup(ranked_lst, ref_lst)

    # returning result
    text_dict[output_key] = ranked_lst
    return text_dict


def sem_train_model(data, cfg):
    if cfg['method'] == 'fasttext':
        txt = cfg["data_path"]
        model = cfg.get('model')
        epoch = cfg.get('epoch', 50)
        thread = cfg.get('thread', 1)
        dim = cfg.get('dim')
        semantic_rank_logger.debug(f"Fasttext Model Training Started")
        model_ = fasttext.train_unsupervised(txt, model=model, epoch=epoch, thread=thread, dim=dim, wordNgrams=3)
        model_.save_model(cfg['model_path'])
        semantic_rank_logger.debug(f"Fasttext Model Training Completed")
    return data


sr_func_map = {
    'semantic_sim_model_init': semantic_sim_model_init,
    'semantic_sim_record': semantic_sim_record,
    'sem_train_model': sem_train_model
}
nlp_func_map.update(sr_func_map)


def test_semantic_sim():
    # To be written
    return None


if __name__ == '__main__':
    test_semantic_sim()
