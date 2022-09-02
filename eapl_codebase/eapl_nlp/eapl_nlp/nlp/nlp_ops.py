import itertools
import spacy
from eapl.df.eapl_kpi import *
import ftfy
from markdownify import markdownify
import html
from bs4 import BeautifulSoup
import json
from string import Template
from time import time
import ast

try:
    from .nlp_glob import *
    from .cfg_templates import nlp_cfg_templates
except ImportError:
    from nlp_glob import *
    from cfg_templates import nlp_cfg_templates

nlp_ops_logger = logging.getLogger(__name__)


def eapl_nlp_pipeline_init(data, cfg):
    nlp_key = cfg.get('nlp_key', 'nlp')
    model = cfg.get('model', "en_core_web_sm")
    nlp = nlp_glob.get(nlp_key, None)
    if nlp is None:
        try:
            if not spacy.util.is_package(model):
                spacy.cli.download(model)
        except BaseException as e:
            nlp_ops_logger.debug(f"Exception: {e}")
            nlp_ops_logger.debug(f"Not able to download the {model}. It may be local package")

        nlp = spacy.load(model)
        nlp_glob[nlp_key] = nlp

    data[nlp_key] = nlp
    return data


def eapl_eval_ops(data, cfg):
    nlp_ops_logger.warning(f"eapl_eval_ops: Shall be Deprecated. Do not use")
    eval_expr = cfg['eval_expr']
    output_key = cfg['output_key']
    tmp_dict = {**data}
    res = eval(eval_expr, tmp_dict)
    data[output_key] = res
    return data


def eapl_record_eval_ops(data, text_dict, op_dict):
    nlp_ops_logger.warning(f"eapl_record_eval_ops: Shall be Deprecated. Do not use")
    eval_expr = op_dict['eval_expr']
    output_key = op_dict['output_key']
    tmp_dict = {**data, **text_dict}
    res = eval(eval_expr, tmp_dict)
    text_dict[output_key] = res
    return text_dict


def eapl_nlp_record_process(data, cfg):
    text_obj_key = cfg.get('text_obj', 'text_obj')
    text_obj = eapl_extract_nested_dict_value(data, text_obj_key)
    ops_ins_beg_key = cfg.get('ops_ins_beg_key', None)
    ops = cfg['ops'].copy()
    if ops_ins_beg_key:
        ops = data[ops_ins_beg_key] + ops
    num_recs = len(text_obj)
    for i, text_dict in enumerate(text_obj):
        nlp_ops_logger.debug(f"eapl_nlp_record_process : Processing record {i + 1}/{num_recs}")
        for op_dict in ops:
            st = time()
            op = op_dict['op']
            if "op_exec_cond" in op_dict:
                nlp_ops_logger.warning(f"eapl_nlp_record_process: To be Deprecated. Do not use op_exec_cond")
            exec_stmt = op_dict.get("op_exec_cond", "True")
            exec_flag = eval(exec_stmt, {}, text_dict)
            if exec_flag:
                func = nlp_func_map[op]
                text_dict = func(data, text_dict, op_dict)
            tt = time() - st
            nlp_ops_logger.debug(f"eapl_nlp_record_process : Processed op: {op} tt: {int(tt * 1000)} ms")

    return data


def eapl_nlp_record_process_l2(data, text_dict, op_dict):
    text_obj_key = op_dict['text_obj']
    dt_type = op_dict.get('dt_type', 'data')
    text_obj = eapl_extract_nested_dict_value(text_dict, text_obj_key)
    dt = data if dt_type == 'data' else text_dict
    ops = op_dict['ops'].copy()
    num_recs = len(text_obj)
    for i, td in enumerate(text_obj):
        nlp_ops_logger.debug(f"eapl_nlp_record_process_l2 : Processing record {i + 1}/{num_recs}")
        for op_dict in ops:
            st = time()
            op = op_dict['op']
            if "op_exec_cond" in op_dict:
                nlp_ops_logger.warning(f"eapl_nlp_record_process_l2: To be Deprecated. Do not use op_exec_cond")
            exec_stmt = op_dict.get("op_exec_cond", "True")
            exec_flag = eval(exec_stmt, {}, td)
            if exec_flag:
                func = nlp_func_map[op]
                td = func(dt, td, op_dict)
            tt = time() - st
            nlp_ops_logger.debug(f"eapl_nlp_record_process_l2 : Processed op: {op} tt: {int(tt * 1000)} ms")

    return text_dict


def eapl_modify_template_str(template_str, subs):
    for k, v in subs.items():
        if not isinstance(v, str):
            template_str = template_str.replace(f"'${{{k}}}'", f'${{{k}}}')
            template_str = template_str.replace(f"'${k}'", f'${k}')
    return template_str


def eapl_data_process_fk_flow(data, config):
    cfg_seq_type = config.get('cfg_seq_type')
    if cfg_seq_type == 'nlp_cfg_template':
        for cfg_template_key in config['config_seq']:
            nlp_cfg = config.get(cfg_template_key, nlp_cfg_templates[cfg_template_key]).copy()
            nlp_cfg['cfg_seq_type'] = None
            eapl_data_process_fk_flow(data, nlp_cfg)

    else:
        nlp_cfg = config
        if cfg_seq_type == 'nlp_cfg_subs_template':
            cfg_tmp = config.copy()
            cfg_subs_template = cfg_tmp['cfg_subs_template']
            subst_key = cfg_tmp.get('subst_key', 'substitutions')
            substitutions = data.get(subst_key, {})
            template_str = str(cfg_subs_template)
            template_str = eapl_modify_template_str(template_str, substitutions)
            template = Template(template_str)
            cfg_tmp = template.substitute(**substitutions)
            nlp_cfg = ast.literal_eval(cfg_tmp)

        for cfg_name in nlp_cfg['config_seq']:
            st = time()
            # nlp_ops_logger.info("Processing Config - %s", cfg_name)
            cfg = nlp_cfg[cfg_name]
            if "cfg_exec_cond" in cfg:
                nlp_ops_logger.warning(f"eapl_data_process_fk_flow: To be Deprecated. Do not use cfg_exec_cond")
            exec_stmt = cfg.get("cfg_exec_cond", "True")
            exec_flag = eval(exec_stmt)
            if not exec_flag:
                continue

            if cfg.get('modify_config', False):
                cfg = eapl_modify_config(nlp_cfg[cfg['ref_config']], cfg['dict_mods'])

            func = cfg['func']
            if isinstance(func, str):
                data = nlp_func_map[func](data, cfg)
            else:
                data = func(data, cfg)
            tt = time() - st
            nlp_ops_logger.info(f"Processed Config - {cfg_name} tt: {int(tt * 1000)} ms")

    nlp_ops_logger.info("Processing End")
    return data


def eapl_create_spacy_doc(nlp, text):
    return nlp(text)


def create_spacy_doc(data, text_dict, op_dict):
    nlp_key = op_dict.get('nlp_key', 'nlp')
    txt_key = op_dict.get('txt_key', 'txt')
    doc_key = op_dict.get('doc_key', 'doc')
    nlp = data[nlp_key]
    text = text_dict[txt_key]
    text_dict[doc_key] = eapl_create_spacy_doc(nlp, text)
    return text_dict


def spacy_doc_tok_proc(data, text_dict, op_dict):
    doc_key = op_dict.get('doc_key', 'doc')
    out_key = op_dict.get('out_key')
    join_flag = op_dict.get('join', False)
    cond_attr = op_dict.get('cond_attr', None)
    cond_attr_match = op_dict.get('cond_attr_match', True)
    cond_neg = op_dict.get("cond_neg", None)
    delim = op_dict.get('delim', " ")
    # Sensible pre_proc_op values: ['lemma_', 'text', 'sents', 'vector', 'vector_norm']
    pre_proc_op = op_dict.get('pre_proc_op', "text")
    doc = text_dict[doc_key]
    if cond_attr:
        if cond_neg:
            toks = [getattr(tok, pre_proc_op) for tok in doc if not getattr(tok, cond_attr) == cond_attr_match]
        else:
            toks = [getattr(tok, pre_proc_op) for tok in doc if getattr(tok, cond_attr) == cond_attr_match]
    else:
        toks = [getattr(tok, pre_proc_op) for tok in doc]

    output = toks
    if join_flag:
        output = delim.join(toks)

    text_dict[out_key] = output
    return text_dict


def spacy_doc_chunk_proc(data, text_dict, op_dict):
    doc_key = op_dict.get('doc_key', 'doc')
    # Sensible chunk_key values: ['ents', 'noun_chunks', 'sents']
    chunk_key = op_dict.get("chunk_key")
    output_key = op_dict.get('out_key')
    out_fmt = op_dict.get('out_fmt', None)
    doc = text_dict[doc_key]
    if out_fmt == 'text':
        chunk_lst = [chunk.text for chunk in getattr(doc, chunk_key)]
    else:
        chunk_lst = [chunk for chunk in getattr(doc, chunk_key)]
    text_dict[output_key] = chunk_lst
    return text_dict


def eapl_str_ops(data, text_dict, op_dict):
    input_key = op_dict.get('input_key')
    output_key = op_dict.get('output_key', input_key)
    txt = text_dict[input_key]
    ops_list = op_dict.get('ops_list')
    for proc_op_dict in ops_list:
        op = proc_op_dict['op']
        # Examples
        # Strip HTML match_pat = r'<.*?>'
        if op == 're_replace':
            p = re.compile(proc_op_dict['match_pat'])
            sub_pat = proc_op_dict['rep_pattern']
            txt = p.sub(sub_pat, txt)

        if op in ['capitalize', 'casefold', 'lower', 'lstrip', 'rstrip', 'strip', 'swapcase', 'title', 'upper']:
            txt = getattr(txt, op)()

        if op == 'fix_text':
            txt = ftfy.fix_text(txt)

        if op == 'markup2markdown':
            txt = markdownify(txt)

        if op == 'limit_text':
            length = proc_op_dict['length']
            txt = txt[:length]

        if op == 'first_char_capitalize':
            txt = txt[0].capitalize() + txt[1:] if (isinstance(txt, str) and len(txt)) else txt

        if op == 'html_to_text':
            elem = BeautifulSoup(txt, features="html.parser")
            text = ''
            for e in elem.descendants:
                if isinstance(e, str):
                    text += e
                elif e.name in ['br', 'p', 'h1', 'h2', 'h3', 'h4', 'tr', 'th']:
                    text += '\n'
                elif e.name == 'li':
                    text += '\n- '

            sents_nls = text.split('\n')
            sents = [f"{sent}." if re.search("[0-9a-zA-Z]+\s*$", sent) else sent for sent in sents_nls]
            txt = "\n".join(sents)

    text_dict[output_key] = txt
    return text_dict


def eapl_str_bool_ops(data, text_dict, op_dict):
    input_key = op_dict.get('input_key')
    output_key = op_dict.get('output_key')
    txt = text_dict[input_key]
    ops_list = op_dict.get('ops_list')
    for proc_op_dict in ops_list:
        op = proc_op_dict['op']
        if op in ['isdigit', 'isalnum', 'isalpha', 'isdecimal', 'isidentifier', 'islower', 'isnumeric', 'isprintable',
                  'isspace', 'isupper']:
            bool_flag = getattr(txt, op)()

    text_dict[output_key] = bool_flag
    return text_dict


def eapl_create_derived_str_td(data, text_dict, op_dict):
    output_key = op_dict['output_key']
    def_map = op_dict.get('def_map', {})
    str_expr = op_dict['str_expr']
    str_expr_type = op_dict.get('str_expr_type', 'format_syn')  # Options: ['format_syn', 'dd_syn']
    remove_nans = op_dict.get('remove_nans', True)

    text_dict_clean = {k: v for k, v in text_dict.items() if v == v} if remove_nans else text_dict
    dd = {**def_map, **text_dict_clean}
    if str_expr_type == 'dd_syn':
        res = eval('f' + repr(str_expr))
    else:
        res = str_expr.format(**dd)
    text_dict[output_key] = res
    return text_dict


def eapl_multikey_processing(data, cfg):
    op = cfg['op']
    op_params = cfg.get('op_params', {})

    if op == 'join_lists':
        join_keys = op_params['join_keys']
        output_key = op_params['output_key']
        out_lst = []
        for key in join_keys:
            out_lst += data[key]
        data[output_key] = out_lst
    elif op == 'ext_nd_value':
        nd_params = op_params['nd_params']
        output_key = op_params['output_key']
        data[output_key] = eapl_extract_nested_dict_value(data, **nd_params)
    elif op == 'dict_from_ext_nd_value':
        nd_params = op_params['nd_params']
        output_key = op_params['output_key']
        output_dict = dict()
        for key, nd_param in nd_params.items():
            output_dict.update({key: eapl_extract_nested_dict_value(data, **nd_param)})
        data[output_key] = output_dict
    elif op == 'create_ld_timestamp':
        input_key = cfg['input_key']
        output_key = cfg.get('output_key', input_key)
        ts_key = cfg['ts_key']
        ts_fmt = cfg.get('ts_fmt', "%d/%m/%Y")
        output = [d.update({ts_key: datetime.datetime.now().strftime(ts_fmt)}) for d in data[input_key]]
        data[output_key] = data[input_key]
    else:
        nlp_ops_logger.debug(f"Currently {op} no supported by eapl_multikey_processing function")

    return data


def eapl_list_slice(data, text_dict, op_dict):
    slice_range = op_dict.get('range', 0)
    lst_key = op_dict.get('input_key')
    lst = text_dict[lst_key]
    out_key = op_dict.get('out_key', lst_key)
    text_dict[out_key] = lst[slice_range]
    return text_dict


def keyword_check(lst_qs, lst_wrds):
    tmp_df = pd.DataFrame({'txt': lst_qs})
    tmp_df = tmp_df[~tmp_df['txt'].isin(lst_wrds)]
    out_lst_qs = tmp_df['txt'].to_list()
    return out_lst_qs


def eapl_drop_string_lst(data, text_dict, op_dict):
    input_key = op_dict.get('input_key', 'txt')
    questions = text_dict[input_key]
    special_kws = op_dict.get('patterns', 'patterns')
    special_kws = eapl_convert_to_list(special_kws)
    questions = eapl_convert_to_list(questions)
    out_key = op_dict.get('out_key', input_key)
    new_lst_qs = keyword_check(questions, special_kws)
    text_dict[out_key] = new_lst_qs

    return text_dict


def manage_data_keys(data, cfg):
    pop_keys = cfg.get('pop_keys', None)
    txt_keys = list(data.keys())
    keep_keys = cfg.get('keep_keys', None)
    rename_keys = cfg.get('rename_keys', None)
    move_keys = cfg.get('move_keys', None)

    pop_key_lst = []
    if keep_keys is not None:
        keep_keys = eapl_convert_to_list(keep_keys)
        pop_key_lst = set(txt_keys) - set(keep_keys)

    if pop_keys:
        pop_keys = eapl_convert_to_list(pop_keys)
        pop_key_lst = set(txt_keys).intersection(set(pop_keys))

    for k in pop_key_lst:
        del data[k]

    if rename_keys:
        for old_k, new_k in rename_keys.items():
            if (old_k in data.keys()) and (new_k not in data.keys()):
                data[new_k] = data.pop(old_k)

    if move_keys:
        for mk_dict in move_keys:
            try:
                sk_list = mk_dict['sk_list']
                dk_list = mk_dict.get('dk_list', [])

                so = data
                for k in sk_list:
                    so = so[k]

                if dk_list:
                    do = data
                    for k in dk_list[:-1]:
                        do = do[k]
                    dst_last_key = dk_list[-1]
                    do[dst_last_key] = so
                else:
                    data.update(so)
            except Exception as e:
                nlp_ops_logger.debug(f"Exception: {e}")
                nlp_ops_logger.debug(f"Failed to move keys for {mk_dict}")

    return data


def manage_text_dict_keys(data, text_dict, op_dict):
    text_dict = manage_data_keys(text_dict, op_dict)

    return text_dict


def qg_replace_substr(data, text_dict, op_dict):
    input_list_key = op_dict.get('input_lst', 'input_lst')
    output_list_key = op_dict.get('output_lst', input_list_key)
    out_lst = text_dict[input_list_key].copy()
    pattern_lst = op_dict['pattern']
    regex_flag = op_dict.get('regex', False)
    rep_pat = op_dict.get('rep_pattern', '')

    if regex_flag:
        for regex_pat in pattern_lst:
            out_lst = [re.sub(regex_pat, rep_pat, txt.strip()).strip() for txt in out_lst]
    else:
        for replace_str in pattern_lst:
            out_lst = [txt.strip().replace(replace_str, rep_pat).strip() for txt in out_lst]
    text_dict[output_list_key] = out_lst
    return text_dict


def qg_split_dedup(data, text_dict, op_dict):
    input_list_key = op_dict.get('input_lst', 'input_lst')
    output_list_key = op_dict.get('output_lst', input_list_key)
    input_lst = text_dict[input_list_key]
    delimiter = op_dict.get('delimiter', None)
    output_list = []
    if delimiter:
        output_list.append([key.split(delimiter) for key in input_lst])
        output_list = list(itertools.chain.from_iterable(output_list))
        output_list = list(itertools.chain.from_iterable(output_list))
        output_list = [output_list[k].strip() for k in range(len(output_list))]
        output_list = list(dict.fromkeys(output_list))

    text_dict[output_list_key] = output_list
    return text_dict


def qg_remove(data, text_dict, op_dict):
    input_list_key = op_dict.get('input_lst', 'input_lst')
    output_list_key = op_dict.get('output_lst', input_list_key)
    method = op_dict.get('method', 'sub_string')
    input_lst = text_dict[input_list_key]
    output_list = input_lst.copy()
    pattern_lst = op_dict['pattern']

    for each_txt in input_lst:
        if method == 'reg_x':
            combined_reg_x = "(" + ")|(".join(pattern_lst) + ")"
            if re.match(combined_reg_x, each_txt):
                output_list.remove(each_txt)
        if method == 'sub_string':
            for rem_str in pattern_lst:
                if rem_str in each_txt:
                    output_list.remove(each_txt)
                    break

    text_dict[output_list_key] = output_list
    return text_dict


# Question check related functions
def is_wh_question_v2(doc):
    wh_tags = ["WDT", "WP", "WP$", "WRB"]
    wh_words = [t for t in doc if t.tag_ in wh_tags]
    sent_initial_is_wh = wh_words and wh_words[0].i == 0
    pied_piped = wh_words and wh_words[0].head.dep_ == "prep"
    pseudocleft = wh_words and wh_words[0].head.dep_ in ["csubj", "advcl"]
    if pseudocleft:
        return False
    return sent_initial_is_wh or pied_piped


def _is_subject(tok):
    subject_deps = {"csubj", "nsubj", "nsubjpass"}
    return tok.dep_ in subject_deps


def is_polar_question_v2(doc):
    root = [t for t in doc if t.dep_ == "ROOT"][0]
    subj = [t for t in root.children if _is_subject(t)]

    if is_wh_question_v2(doc):
        return False
    aux = [t for t in root.lefts if t.dep_ == "aux"]
    if subj and aux:
        return aux[0].i < subj[0].i
    root_is_inflected_copula = root.pos_ == "VERB" and root.tag_ != "VB"
    if subj and root_is_inflected_copula:
        return root.i < subj[0].i

    return False


def get_question_type(nlp, sent):
    sent_is_q = False
    doc = nlp(sent)
    is_wh = is_wh_question_v2(doc)
    is_polar = is_polar_question_v2(doc)
    has_q_mark = '?' in sent
    if is_wh or is_polar or has_q_mark:
        sent_is_q = True
    return sent_is_q


def qg_question_check(data, text_dict, op_dict=None):
    nlp_key = op_dict.get('nlp_key', 'nlp')
    input_key = op_dict.get('input_key', 'input_lst')
    output_key = op_dict.get('output_key', input_key)
    output_fmt = op_dict.get('output_fmt', None)
    ques_key = op_dict.get('ques_key', 'is_question')
    q_str_lst = eapl_convert_to_list(text_dict[input_key])
    nlp = data[nlp_key]
    output = q_str_lst
    text = q_str_lst[-1] if q_str_lst else ""
    is_ques = False
    if len(text.strip()) > 0:
        is_ques = get_question_type(nlp, text)
        if is_ques:
            output = text

    if output_fmt == 'list':
        output = eapl_convert_to_list(output)

    text_dict[output_key] = output
    text_dict[ques_key] = is_ques
    return text_dict


def eapl_create_unq_id(data, cfg):
    from uuid import uuid4
    import mmh3
    text_obj_key = cfg['text_obj_key']
    ref_key = cfg.get('ref_key', None)
    out_id_key = cfg["out_id_key"]
    method = cfg.get("method", "random")
    out_key = cfg.get("out_key", text_obj_key)
    join_str = cfg.get("join_str", "_")
    id_hash_keys = cfg.get("id_hash_keys", [])

    tmp_df = pd.DataFrame(data[text_obj_key])
    if method == 'index_id':
        tmp_df = tmp_df.reset_index().rename(columns={"index": out_id_key})
    elif method == 'random':
        uuids = [str(uuid4()) for i in range(tmp_df.shape[0])]
        tmp_df[out_id_key] = uuids
    elif method == 'reference':
        tmp_df = tmp_df.reset_index()
        tmp_df[out_id_key] = tmp_df.apply(lambda x: f"{x[ref_key]}{join_str}{x['index']}", axis=1)
        tmp_df.drop(columns=['index'], inplace=True)
    elif method == 'hash':
        tmp_strng5 = tmp_df[id_hash_keys].astype(str).apply(lambda x: f' {join_str} '.join(x), axis=1)
        tmp_df[out_id_key] = tmp_strng5.apply(lambda x: '{:02x}'.format(mmh3.hash128(x, signed=False)))
    data[out_key] = tmp_df.to_dict('records')
    return data


def eapl_concat_kv(data, text_dict, op_dict):
    input_key = op_dict['input_key']
    concat_keys = op_dict['concat_keys']
    output_key = op_dict.get('output_key', input_key)
    start_str = op_dict.get("start_str", "?")
    join_str = op_dict.get("join_str", "&")
    kv_join_str = op_dict.get("kv_join_str", "=")
    dict_source = op_dict.get("dict_source", "data")
    txt = text_dict[input_key]

    concat_keys = eapl_convert_to_list(concat_keys)
    dct = data if dict_source == "data" else text_dict
    kv_list = []
    for k in concat_keys:
        kv_list.append(f"{k}{kv_join_str}{dct[k]}")

    txt = txt + start_str + join_str.join(kv_list)
    text_dict[output_key] = txt
    return text_dict


def eapl_pad_strings(data, cfg):
    input_key = cfg['input_key']
    output_key = cfg.get('output_key', input_key)
    batch_size = cfg.get('batch_size', 5)
    str_key = cfg['str_key']
    pad_char = cfg.get('pad_char', ' ')  # normal white space character
    eos_char = cfg.get('eos_char', '⠀')  # invisible white space character
    fillna_val = cfg.get('fillna_val', '')
    width_factor = cfg.get('width_factor', 1.0)
    recs = data[input_key]
    df = pd.DataFrame(recs)
    df[str_key] = df[str_key].fillna(value=fillna_val)
    df_cols = list(df.columns)

    lk, gidk, gid_mlk = f"{str_key}_len", f"{str_key}_gid", f"{str_key}_gid_ml"
    ldk, ld_wfk, ps = f"{str_key}_len_diff", f"{str_key}_len_diff_wf", f"{str_key}_pad_str"
    df[lk] = df[str_key].str.len()
    df[gidk] = df.index // batch_size
    max_df = df.groupby(gidk)[lk].max().reset_index()
    max_df.rename(columns={lk: gid_mlk}, inplace=True)

    df = df.merge(max_df, how='left', on=gidk)
    df[ldk] = df[gid_mlk] - df[lk]
    df[ld_wfk] = (df[ldk] * width_factor).astype(int)
    df[ps] = df.apply(lambda x: (pad_char * x[ld_wfk] + eos_char)[-x[ld_wfk]:], axis=1)
    df[ps] = np.where(df[ld_wfk] > 0, df[ps], '')
    df[str_key] = df.apply(lambda x: x[str_key] + x[ps], axis=1)

    recs = df[df_cols].to_dict(orient='records')
    data[output_key] = recs
    return data

from .nlp_ent_extraction import rake_init
nlp_ops_funcs = {
    'eapl_nlp_pipeline_init': eapl_nlp_pipeline_init,
    'eapl_eval_ops': eapl_eval_ops,
    'eapl_record_eval_ops': eapl_record_eval_ops,
    'eapl_nlp_record_process': eapl_nlp_record_process,
    'eapl_nlp_record_process_l2': eapl_nlp_record_process_l2,
    'eapl_data_process_fk_flow': eapl_data_process_fk_flow,
    'create_spacy_doc': create_spacy_doc,
    'replace_substr': qg_replace_substr,
    'split_dedup': qg_split_dedup,
    'remove': qg_remove,
    'question': qg_question_check,
    'manage_text_dict_keys': manage_text_dict_keys,
    'manage_data_keys': manage_data_keys,
    'spacy_doc_tok_proc': spacy_doc_tok_proc,
    'spacy_doc_chunk_proc': spacy_doc_chunk_proc,
    'eapl_str_ops': eapl_str_ops,
    'eapl_str_bool_ops': eapl_str_bool_ops,
    'eapl_create_derived_str_td': eapl_create_derived_str_td,
    'eapl_multikey_processing': eapl_multikey_processing,
    'eapl_concat_kv': eapl_concat_kv,
    'eapl_pad_strings': eapl_pad_strings,
    'eapl_create_unq_id': eapl_create_unq_id,
    'rake_init': rake_init
}
nlp_func_map.update(nlp_ops_funcs)

from .simpplr_cfg_templates import simpplr_config_api
nlp_func_map['simpplr_config_api'] = simpplr_config_api

from .simpplr_mongodb_connection import mongodb_connect
nlp_func_map['mongodb_connect'] = mongodb_connect

from .simpplr_content_reco import simpplr_data_harmonisation
nlp_func_map['simpplr_data_harmonisation'] = simpplr_data_harmonisation

def test_nlp_ops():
    from pprint import pprint
    data = {
        'text_ld': [
            {
                'source': 'a'
            },
            {
                'source': 'b'
            },
            {
                'source': 'a'
            },
        ]
    }

    nlp_cfg = {
        'config_seq': ['data_eval'],
        'data_eval': {
            'func': 'eapl_create_unq_id',
            "text_obj_key": 'text_ld',
            "method": "reference",
            "out_id_key": "id",
            "ref_key": "source"
        },
    }
    pprint(data['text_ld'])
    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['text_ld'])


def test_create_id():
    from pprint import pprint
    data = {
        "text_ld": [
            {
                "source": "a",
                "tag": 1
            },
            {
                "source": "a",
                "tag": 2
            }
        ]
    }

    nlp_cfg = {
        "config_seq": ["id_hash_creation"],
        "id_hash_creation": {
            "func": "eapl_create_unq_id",
            "text_obj_key": "text_ld",
            "method": "hash",
            "id_hash_keys": ["source", "tag"],
            "out_id_key": "id"

        }

    }
    pprint(data['text_ld'])
    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['text_ld'])


if __name__ == '__main__':
    # test_nlp_ops()
    test_create_id()
