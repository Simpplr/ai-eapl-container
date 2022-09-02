import logging
from copy import deepcopy
import pysolr

try:
    from .nlp_glob import *
    from .nlp_utils import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_utils import *
    from nlp_ops import nlp_ops_funcs

search_logger = logging.getLogger(__name__)


def eapl_nlp_search(data, cfg):
    url = cfg.get('search_url', 'https://emma-sap-ps.emplay.net/search/nlp-solr-search/')
    sr_key = cfg.get('sr_key', 'search_results')
    q_key = cfg.get('q_key', 'query')
    query = data[q_key]

    user_profile_key = cfg.get('user_profile_key', 'user_profile')
    user_profile = data.get(user_profile_key, {})
    if not user_profile:
        user_profile = {}

    search_params = cfg['search_params'].copy()
    search_params['query'] = query
    search_params['user_profile'] = user_profile

    r = requests.post(url=url, json=search_params)

    if r.status_code == 200:
        data[sr_key] = json.loads(r.content)
    else:
        data[sr_key] = {}

    return data


def eapl_solr_luis_params(data, cfg):
    entities_key = cfg.get('entities_key', "user_profile|luis_recognizer_result|entities")
    fq_keys = cfg.get("fq_keys", None)
    fq_dct_key = cfg.get("fq_dct_key", "fq_dct_params")
    fq_key = cfg.get("fq_key", "fq")
    join_str = cfg.get("join_str", " OR ")
    include_key = cfg.get("include_key", True)
    entities_dict = eapl_extract_nested_dict_value(data, entities_key, def_val={})
    fq_dict = dict()
    fq_val_list = []
    if fq_keys:
        fq_luis_keys = list(set(fq_keys).intersection(entities_dict.keys()))
        search_logger.debug(f"fq_luis_keys : {fq_luis_keys}")
        for key in fq_luis_keys:
            ent_val_list = []
            for ent in entities_dict[key]:
                fq_val_tmp = f'{key}:"{ent[0]}"' if include_key else f'{ent[0]}'
                if fq_val_tmp not in ent_val_list:
                    ent_val_list.append(fq_val_tmp)

            fq_val = join_str.join(ent_val_list)
            fq_val_list.append(fq_val)
        fq_dict[fq_key] = fq_val_list

    search_logger.debug(f"fq_dict : {fq_dict}")
    data[fq_dct_key] = fq_dict
    return data


def eapl_solr_highlight(data, cfg):
    sr_key = cfg.get('sr_key', 'search_results')
    search_output = data[sr_key]
    hl_output = pd.DataFrame(search_output.highlighting).T.applymap(lambda x: "".join(x))
    hl_output.rename(columns=lambda x: "hl_" + x, inplace=True)
    hl_output['id'] = hl_output.index
    doc_output = pd.DataFrame(search_output.docs)
    search_output.docs = doc_output.merge(hl_output, on="id").to_dict('records')
    data[sr_key] = search_output

    return data


def eapl_filt_records_fq_params(data, cfg):
    fq_dct_key = cfg.get("fq_dct_key", "fq_dct_params")
    sr_key = cfg.get('sr_key', 'search_results')

    fq_list = data[fq_dct_key].get('fq', None)
    if fq_list:
        df = pd.DataFrame.from_records(data[sr_key])
        fq_val = [f"({cond})" for cond in fq_list]
        fq_filt_cond = " and ".join(fq_val).replace(":", "==").replace(" OR ", " or ")
        df = df.query(fq_filt_cond)
        data[sr_key] = df.to_dict(orient='records')

    return data


def eapl_remove_entity(data, cfg):
    query_key = cfg.get("query_key", "query")
    entities_key = cfg.get('entities_key', "user_profile|luis_recognizer_result|entities|$instance")
    patterns_lst = cfg.get("patterns_lst", [])
    stopword_lst = cfg.get("stopword_lst", []).copy()
    outkey = cfg.get("out_key", query_key)
    all_stopwords = ["'d", "'ll", "'m", "'re", "'s", "'ve", 'a', 'about', 'all', 'almost', 'alone', 'along', 'already',
                     'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'any', 'anyhow', 'anyone',
                     'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'be', 'became', 'because', 'become',
                     'becomes', 'becoming', 'been', 'being', 'can', 'cannot', 'could', 'did', 'do', 'does', 'doing',
                     'done', 'else', 'elsewhere', 'for', 'from', 'had', 'has', 'have', 'he', 'hence', 'her', 'here',
                     'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                     'however', 'i', 'if', 'in', 'is', 'its', 'itself', 'may', 'me', 'my', 'myself', "n't",
                     'of', 'often', 'on', 'our', 'ours', 'ourselves', 'she', 'should', 'so', 'than', 'that', 'the',
                     'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
                     'therein', 'thereupon', 'these', 'they', 'this', 'those', 'though', 'to', 'was', 'we',
                     'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
                     'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'who', 'whoever',
                     'whom', 'whose', 'why', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours',
                     'yourself', 'yourselves']

    stopword_lst.extend(all_stopwords)
    query = data[query_key]
    org_query = query
    try:
        entities_dict = eapl_extract_nested_dict_value(data, entities_key, def_val={})
        tmp_rmv = []
        for each_meta_data in list(set(entities_dict.keys()).intersection(set(patterns_lst))):
            meta_dict = entities_dict[each_meta_data]
            tmp_rmv.extend(pd.DataFrame(meta_dict)['text'].to_list())

        tmp_rmv = sorted(tmp_rmv, key=len, reverse=True)
        for i in tmp_rmv:
            query = query.replace(i, "")

        query = " ".join([word for word in query.split(" ") if word not in stopword_lst])

    except Exception as e:
        query = org_query
        search_logger.debug(f"Exception: {e}")
        search_logger.debug(f"No entities removed for : {query}")

    if query.strip() == "":
        query = org_query

    data[outkey] = query
    search_logger.debug("The final query : {query}" )
    return data


def eapl_solr_luis_bq_params(data, cfg):
    entities_key = cfg.get('entities_key', "user_profile|luis_recognizer_result|entities")
    bq_keys = cfg.get("bq_keys", None)
    bq_dct_key = cfg.get("bq_dct_key", "bq_dct_params")
    bq_static_bst = cfg.get("bq_static", [])
    bq_key = cfg.get("bq_key", "bq")
    include_key = cfg.get("include_key", True)
    entities_dict = eapl_extract_nested_dict_value(data, entities_key, def_val={})
    bq_dict = dict()
    bq_val_list = []
    if bq_keys:
        bq_luis_keys = list(set(bq_keys.keys()).intersection(entities_dict.keys()))
        for key in bq_luis_keys:
            ent_val_list = []
            for ent in entities_dict[key]:
                bq_val_tmp = f'{key}:{ent[0]}^{bq_keys[key]}' if include_key else f'{ent[0]}^{bq_keys[key]}'
                if bq_val_tmp not in ent_val_list:
                    bq_val_list.append(bq_val_tmp)

        bq_dict[bq_key] = bq_val_list

    if bq_static_bst:
        bq_dict[bq_key].extend(bq_static_bst)

    data[bq_dct_key] = bq_dict
    return data


def eapl_solr_search(data, cfg):
    solr_url_key = cfg['solr_url_key']
    solr_url = cfg['solr_url']
    sr_key = cfg.get('sr_key', 'search_results')
    query_key = cfg.get('query_key', 'query')
    query = data[query_key]
    cfg_solr_params = cfg.get('cfg_solr_params', dict()).copy()
    data_solr_params_key = cfg.get('data_solr_params_key', None)
    fq_keys = cfg.get("fq_keys", None)
    bq_keys = cfg.get("bq_keys", None)

    if data_solr_params_key:
        data_solr_params = data[data_solr_params_key]
        cfg_solr_params.update(data_solr_params)

    if fq_keys:
        data = eapl_solr_luis_params(data, cfg)
        fq_dct_key = cfg.get("fq_dct_key", "fq_dct_params")
        cfg_solr_params.update(data[fq_dct_key])

    if bq_keys:
        data = eapl_solr_luis_bq_params(data, cfg)
        bq_dict_key = cfg.get("bq_dct_key", "bq_dct_params")
        cfg_solr_params.update(data[bq_dict_key])

    if solr_url_key in nlp_glob:
        solr_obj = nlp_glob[solr_url_key]
    else:
        solr_obj = pysolr.Solr(solr_url)
        nlp_glob[solr_url_key] = solr_obj

    data[sr_key] = solr_obj.search(query, **cfg_solr_params)

    try:
        if cfg_solr_params.get("hl", "off") == "on":
            eapl_solr_highlight(data, cfg)

        data[sr_key] = data[sr_key].docs

    except:
        data[sr_key] = []

    return data

def eapl_search_result_reformat_cai(data, cfg):
    sr_key = cfg.get('sr_key', "searchResults")
    search_res = eapl_extract_nested_dict_value(data, sr_key)
    out_key = cfg.get('out_key', 'reformatted_res')
    error_msg = cfg.get('error_msg',
                        {"type": "text", "content": "Sorry! I couldn't find an answer. Please try different query",
                         "markdown": "true", "delay": "1"})
    def_carousal = {
        "type": "carousel",
        "delay": 1,
        "content": [
            {
                "title": "question",
                "description": "Content_Text",
                "buttons": [
                    {
                        "title": "Click Here",
                        "value": "postback",
                        "type": "web_url"
                    }
                ]
            }
        ]
    }
    carousel = cfg.get('carousel', def_carousal)
    temp_carousel = carousel.copy()
    content = deepcopy(temp_carousel['content'][0])
    del content['buttons']
    button = deepcopy(temp_carousel['content'][0]['buttons'])
    button_value = button[0]["value"]
    content_response = []
    for sr in search_res:
        cai_resp_dict = {}
        for key, value in content.items():
            cai_resp_dict[key] = sr.get(value, value)
        cai_resp_dict.update({"buttons": deepcopy(button)})
        cai_resp_dict["buttons"][0]["value"] = sr.get(button_value, button_value)
        content_response.append(cai_resp_dict)

    if len(content_response) > 0:
        carousel['content'] = content_response
    else:
        carousel = error_msg

    data[out_key] = carousel
    return data


nlp_search_fmap = {
    "eapl_nlp_search": eapl_nlp_search,
    "eapl_solr_search": eapl_solr_search,
    "eapl_solr_luis_params": eapl_solr_luis_params,
    "eapl_solr_luis_bq_params": eapl_solr_luis_bq_params,
    "eapl_remove_entity": eapl_remove_entity,
    "eapl_filt_records_fq_params": eapl_filt_records_fq_params,
    "eapl_search_result_reformat_cai": eapl_search_result_reformat_cai
}
nlp_func_map.update(nlp_search_fmap)


def test_nlp_search():
    from pprint import pprint
    nlp_cfg = {
        "config_seq": ["import_funcs", "gs_kws_extract", "update_query", "solr_search"],
        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "eapl_google_search": "from web_search import web_search_fmap"
            }
        },
        "gs_kws_extract": {
            'func': 'eapl_google_search',
            'search_method': 'googlesearch_custom',
            'sr_key': 'gs_search_res',
            'sm_params': {
                'cx': 'f95763bbfd8a2c07f',  # help.sap.com pages
                'key': 'AIzaSyBxT3h0irlSwhLm3DjJTL16VVTgVFwgsuY',  # bot-admin
                'num': 10,
            },
        },
        "update_query": {
            "func": "eapl_eval_ops",
            "eval_expr": "query + ' ' + ' '.join([f'\"{kw}\"' for kw in hl_kws])",
            "output_key": "query"
        },
        "solr_search": {
            "func": "eapl_solr_search",
            'solr_url_key': 'test_solr_url',
            "solr_url": "http://emma-sap-ps.emplay.net:8987/solr/empsapio_1",
            "sr_key": "search_results",
            "entities_key": "user_profile|luis_recognizer_result|entities",
            "fq_keys": ["source"],
            "query_key": "query",
            "cfg_solr_params": {
                'fl': ['question', 'Page_No', 'source', 'Content_URL_buttons'],
                'qf': ['Content_Title^20', 'question^10', 'Title_Hierarchy^5', 'Content_Text^2', '_text_^1'],
                'defType': 'edismax',
                'rows': 2
            },
            "data_solr_params_key": "dt_solr_params"
        },
    }

    data = {
        "query": "performance management template",
        "dt_solr_params": {
            'hl.fragsize': 20,
            'fq': ['source:"Performance Management" OR source:"Goal Management" OR source:"Calibration"']
        },
        "user_profile": {
            "luis_recognizer_result": {
                "text": "Direct link to specific performance form in Performance Management",
                "entities": {
                    "source": [
                        [
                            "Performance Management"
                        ],
                        [
                            "Goal Management"
                        ]
                    ],
                },
            }
        }
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)

    return None


if __name__ == '__main__':
    test_nlp_search()
