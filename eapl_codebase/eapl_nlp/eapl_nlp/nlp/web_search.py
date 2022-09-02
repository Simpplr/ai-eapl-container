import os
import logging
import requests
import json
import re
from collections import Counter
from googlesearch import search
import numpy as np
import pandas as pd
from pprint import pprint

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
    from .nlp_utils import nlp_utils_fmap
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs
    from nlp_utils import nlp_utils_fmap

logger = logging.getLogger(__name__)


def eapl_get_highlighted_keywords(search_res_dict):
    kws_list = []
    rem_kws_list = ['...', '']
    for sr in search_res_dict:
        cc_snippet = sr['htmlTitle'] + " " + sr['htmlSnippet']
        cc_snippet = re.sub(r"</b>\s+<br>\n<b>", ' ', cc_snippet)
        cc_snippet = re.sub(r"</b>\s*<br>\n<b>", '', cc_snippet)
        kws = re.findall("<b>(.+?)</b>", cc_snippet)
        kws_list = kws_list + kws

    kws_list = [kw.lower() for kw in kws_list if kw not in rem_kws_list]
    kws_dict = Counter(kws_list)
    kws_list = list(set(kws_list))
    return kws_list, kws_dict


def eapl_append_kws_to_search(query, kws_dict):
    kws_type = kws_dict.get('kws_type', 'list')
    keywords = kws_dict['keywords']

    if kws_type == 'list':
        for kw in keywords:
            if kw != '':
                query = f'{query} "{kw}"'

    logger.debug(f"After appending keywords: {query}")
    return query


def eapl_get_top_score_intent_luis(intents_dict, intent_fallback_params):
    id_cp = intents_dict.copy()
    id_cp.update(intent_fallback_params)
    return max(id_cp, key=lambda x: id_cp[x]['score'])


def eapl_googlesearch_custom(data, cfg, query_str):
    sm_params = cfg.get('sm_params', {}).copy()
    data_gs_params_key = cfg.get("data_gs_params_key", None)
    if data_gs_params_key:
        sm_params.update(eapl_extract_nested_dict_value(data, data_gs_params_key, delim="|", def_val={}))

    key = sm_params.get('key')
    cx = sm_params.get('cx')
    num = sm_params.get('num', 10)
    endpoint = sm_params.get('endpoint', 'https://www.googleapis.com/customsearch/v1?')
    intent_to_cx = cfg.get('intent_to_cx', False)
    corrected_query_key = cfg.get("corrected_query_key", "corrected_query")
    eapl_gs_cols = cfg.get("eapl_gs_cols", ["title", "htmlTitle", "link", "snippet", "htmlSnippet"])

    if intent_to_cx:
        intents_key = cfg.get('intents_key', "user_profile|luis_recognizer_result|intents")
        intents_dict = eapl_extract_nested_dict_value(data, intents_key)
        intent_to_cx_params = cfg.get('intent_to_cx_params')
        if intents_key:
            intent_fallback_params = cfg.get("intent_fallback_params", {"None": {'score': 0.3}})
            top_intent = eapl_get_top_score_intent_luis(intents_dict, intent_fallback_params)
            cx = intent_to_cx_params[top_intent]

    params = (
        ('q', query_str),
        ('key', key),
        ('cx', cx),
        ('num', num)
    )
    parsed, search_res = dict(), []
    try:
        response = eapl_req_sess.get(endpoint, params=params)
        parsed = json.loads(response.content)
        search_res = parsed['items']
    except:
        logger.debug(f"eapl_googlesearch_custom failed to search: {query_str}")
        logger.debug(f"Error Response: \n {parsed}")

    search_res_df = pd.DataFrame(search_res, columns=eapl_gs_cols).fillna("")
    search_res = search_res_df.to_dict(orient='records')

    data[corrected_query_key] = query_str
    if parsed.get('spelling', None):
        data[corrected_query_key] = parsed['spelling']['correctedQuery']
        logger.debug(f"Spellchecked corrected Query: {data[corrected_query_key]}")

    return search_res


def eapl_googlesearch_public(query_str, cfg):
    sm_params = cfg.get('sm_params', {})
    num = sm_params.get("num", 5)
    start = sm_params.get("start", 0)
    stop = sm_params.get("stop", num)
    pause = sm_params.get("pause", 2)
    extra_params = sm_params.get("extra_params", None)

    if os.path.exists(".google-cookie"):
        os.remove(".google-cookie")
    search_res = search(query_str, num=num, start=start, stop=stop, pause=pause, extra_params=extra_params)
    search_res_ld = [{'link': link} for link in search_res]
    return search_res_ld


def eapl_google_search(data, cfg):
    query_key = cfg.get('query_key', 'query')
    site_key = cfg.get('site_key', None)
    sr_key = cfg.get('sr_key', 'search_results')
    hl_kws_key = cfg.get('hl_kws_key', None)
    hl_kws_type = cfg.get('hl_kws_type', 'list')
    search_phrase_key = cfg.get('search_phrase_key', None)
    search_kws_key = cfg.get('search_kws_key', None)
    search_method = cfg.get('search_method', 'googlesearch_public')
    rep_comma_by_or = cfg.get('rep_comma_by_or', False)
    corrected_query_key = cfg.get('corrected_query_key', "corrected_query")
    cse_api_second_call = cfg.get('cse_api_second_call', False)
    query = data[query_key]
    logger.debug(f"Original Search Query: {query}")
    if search_phrase_key:
        try:
            ent_query = eapl_extract_nested_dict_value(data, search_phrase_key)[0]
            if len(ent_query) > 0:
                query = ent_query
                logger.debug(f"NLU parsed Search Query: {query}")
        except:
            logger.debug(f"No Search phrase detected from NLU recognizer: {query}")

    query_str = query
    if isinstance(query, list):
        query_str = " OR ".join(query)

    if rep_comma_by_or:
        query_str = query_str.replace(",", " OR ")

    if search_kws_key:
        search_kws_dict = eapl_extract_nested_dict_value(data, search_kws_key)
        query_str = eapl_append_kws_to_search(query_str, search_kws_dict)

    if site_key:
        site = data[site_key]
        site_str = f"site:{site}" if isinstance(site, str) else " OR ".join([f"site:{s}" for s in site])
        query_str = f"{query_str} {site_str}"

    logger.debug(f"Search Query: {query_str}")
    if search_method == 'googlesearch_public':
        search_res = eapl_googlesearch_public(query_str, cfg)
    elif search_method == 'googlesearch_custom':
        search_res = eapl_googlesearch_custom(data, cfg, query_str)
        if data[corrected_query_key] != query_str and cse_api_second_call:
            search_res = eapl_googlesearch_custom(data, cfg, data[corrected_query_key])
    if hl_kws_key:
        hl_kws, kws_dict = eapl_get_highlighted_keywords(search_res)
        if hl_kws_type == 'dict':
            hl_kws = kws_dict
        data[hl_kws_key] = hl_kws

    logger.debug(f"Search URLs found: \n{search_res}")
    data[sr_key] = search_res
    return data


def eapl_google_search_td(data, text_dict, cfg):
    return eapl_google_search(text_dict, cfg)


def parse_url(url, rm_patterns=[]):
    url_split = url.split('/')
    last = url_split.pop(-1).split('.htm')[0]
    url_split.append(last)

    match_list = [part for part in url_split if part not in rm_patterns]

    return match_list


def eapl_match_search_data(data, cfg):
    sr_key = cfg.get('sr_key', 'search_results')
    ref_df_key = cfg.get('ref_df_key', 'ref_df')
    out_key = cfg.get('out_key', 'match_ld')
    url_key = cfg.get('url_key', 'link')
    rank_key = cfg.get('rank_key', 'rank')
    url_match_key = cfg.get('url_match_key', 'url_match_key')
    rm_patterns_key = cfg.get('rm_patterns_key', 'rm_patterns')
    rm_patterns = set(data.get(rm_patterns_key, []))
    ref_df = data.get(ref_df_key, None)
    out_cols = cfg.get("out_cols", None)
    matched_first = cfg.get("matched_first", True)
    nmr_mapping_cols = cfg.get("nmr_mapping_cols", None)  # non matched response for column copying
    nmr_mapping_vals = cfg.get("nmr_mapping_vals", [])  # non matched response default value
    unq_keys = cfg.get("unq_keys")

    search_results = data[sr_key]
    sr_df = pd.DataFrame(search_results)
    sr_df[rank_key] = range(sr_df.shape[0])
    matched_df = sr_df
    if isinstance(ref_df, pd.DataFrame) and sr_df.shape[0] > 0:
        sr_df[url_match_key] = sr_df[url_key].apply(parse_url, rm_patterns=rm_patterns)
        sr_clean_df = sr_df.explode(url_match_key, ignore_index=True).drop_duplicates(subset=url_match_key)

        # Support when no match is found in the product table
        if nmr_mapping_cols:
            joined_df = sr_clean_df.merge(ref_df, how='left', on=url_match_key, indicator=True)
            for dst_col, src_col in nmr_mapping_cols:
                true_cond_ser = joined_df.eval(f"{dst_col} != {dst_col}")
                joined_df[dst_col] = np.where(true_cond_ser, joined_df[src_col], joined_df[dst_col])
            for dst_col, val in nmr_mapping_vals:
                true_cond_ser = joined_df.eval(f"{dst_col} != {dst_col}")
                joined_df[dst_col] = np.where(true_cond_ser, val, joined_df[dst_col])

            if matched_first:
                matched_df = joined_df.query(f"_merge != 'left_only'")
                search_only_df = joined_df.query(f"_merge == 'left_only'")
                matched_df = pd.concat([matched_df, search_only_df], axis=0)
            else:
                matched_df = joined_df

        else:
            matched_df = sr_clean_df.merge(ref_df, how='inner', on=url_match_key)

    if out_cols:
        out_cols = set(out_cols).intersection(set(matched_df.columns))
    else:
        out_cols = matched_df.columns

    if unq_keys:
        matched_df = matched_df.drop_duplicates(subset=unq_keys)
    recs = matched_df[out_cols].to_dict(orient='records')
    data[out_key] = recs
    return data


web_search_fmap = {
    'eapl_google_search': eapl_google_search,
    'eapl_match_search_data': eapl_match_search_data
}
nlp_func_map.update(web_search_fmap)


def test_gs_custom_query():
    data = {
        "query": 'user productivity',
        "user_profile": {
            "gs_params": {
                "cx": "a8b6cf7dfb035a670",
                "key": "AIzaSyDdIigfyGpC4e0-JQdPcX5ERux71bDFrBE",
                "num": 10
            },
            "luis_recognizer_result": {
                "topIntent": "search",
                "intents": {
                    "search": {
                        "score": 0.99999
                    },
                    "None": {
                        "score": 3.356505E-05
                    },
                    "welcome": {
                        "score": 1.8888853E-06
                    }
                },
                "entities": {
                    "l3_product": [
                        [
                            "Performance Management"
                        ],
                        [
                            "Onboarding"
                        ]
                    ]

                }
            }
        }
    }

    nlp_cfg = {
        'config_seq': ['import_funcs', 'load_prod_data', "create_kws", 'google_search_custom', 'prod_match',
                       'manage_output'],

        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "eapl_nlp_search": "from eapl_nlp_search import nlp_search_fmap"
            }
        },

        'load_prod_data': {
            'func': 'eapl_read_file_data',
            'read_method': 'pd_read_csv',
            'file_or_url_path': 'https://s3.amazonaws.com/emplay.botresources/product_finder/a_z_prods_cleandummy.csv',
            'out_key': 'ref_df'

        },

        "create_kws": {
            "func": "eapl_solr_luis_params",
            "join_str": '" OR "',
            "fq_dct_key": "gs_kws_dct",
            "fq_keys": ["source"],
            "fq_key": "keywords",
            "include_key": False
        },

        'google_search_custom': {
            'func': 'eapl_google_search',
            'search_method': 'googlesearch_custom',
            "search_kws_key": "gs_kws_dct",
            "data_gs_params_key": "user_profile|gs_params",
            # "hl_kws_key": "hl_kws",
            'sr_key': 'search_results',
            'sm_params': {
                # "cx": "a8b6cf7dfb035a670",
                # "key": "AIzaSyDdIigfyGpC4e0-JQdPcX5ERux71bDFrBE",
                # "num": 10

            },
            'cse_api_second_call': False
        },

        'prod_match': {
            'func': 'eapl_match_search_data',
            'out_key': 'searchResults',
            'out_cols': ["title", "htmlTitle", "link", "displayLink", "formattedUrl",
                         "htmlFormattedUrl", "snippet", "htmlSnippet", 'icon_url',
                         'product_name', 'url',
                         'video_url', 'customer_stories_top_url', 'trial_link', 'request_demo_link'],
            'nmr_mapping_cols': [['url', 'link'], ['description', 'snippet'], ['product_name', 'title']],
            'nmr_mapping_vals':
                [['icon_url',
                  'https://www.sap.com/content/dam/application/imagelibrary/pictograms/281000/281230-pictogram-purple.svg'
                  ],
                 [
                     "video_url",
                     "https://s3.amazonaws.com/emplay.botresources/product_finder/BTP.mp4"
                 ],
                 [
                     "customer_stories_top_url",
                     "www.sap.com/documents/2021/01/045d12e7-c77d-0010-87a3-c30de2ffd8ff.html"
                 ],
                 [
                     "trial_link",
                     "https://www.sap.com/cmp/td/sap-master-data-governance-trial.html"
                 ],
                 [
                     "request_demo_link",
                     "https://www.sap.com/cmp/dg/database-management-demo/index.html"
                 ]
                 ],
            'unq_keys': 'url',
            'matched_first': True
        },

        'manage_output': {
            'func': 'manage_data_keys',
            'keep_keys': ['searchResults', 'corrected_query']

        }
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)


if __name__ == '__main__':
    logging.basicConfig(
        level='DEBUG',
        format="%(asctime)s %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler('nlp_pipeline.log', mode='w'),
            logging.StreamHandler()
        ])
    test_gs_custom_query()
