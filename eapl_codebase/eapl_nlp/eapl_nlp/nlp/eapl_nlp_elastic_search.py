import logging
from elasticsearch import Elasticsearch

try:
    from .nlp_glob import *
    from .nlp_utils import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_utils import *
    from nlp_ops import nlp_ops_funcs

es_logger = logging.getLogger(__name__)


def eapl_hs_init_es(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    docstore_key = cfg.get('docstore_key', 'docstore')
    es_obj_key = cfg.get("es_obj_key", "es_obj")
    hs_obj = data[hs_obj_key]

    if es_obj_key in nlp_glob:
        data[es_obj_key] = nlp_glob[es_obj_key]
    else:
        docstore = hs_obj[docstore_key]
        es_obj = docstore.client
        data[es_obj_key] = es_obj
        nlp_glob[es_obj_key] = es_obj

    return data


def eapl_es_op_cal(data, cfg):
    es_obj_key = cfg.get("es_obj_key", "es_obj")
    api_cal = cfg.get("api_cal", "search")
    api_cal_params = cfg.get("api_cal_params", {})
    es_out = cfg.get("es_out", "es_res")

    es = data[es_obj_key]

    try:
        for func in api_cal.split("|"):
            es = getattr(es, func)
        es_res = es(**api_cal_params)

        if 'hits' in es_res.keys():
            if 'hits' in es_res['hits'].keys():
                data[es_out] = es_res['hits']['hits']
        else:
            data[es_out] = es_res

    except:
        data[es_out] = []

    return data


nlp_elastic_search_fmap = {
    "eapl_es_op_cal": eapl_es_op_cal,
    "eapl_hs_init_es": eapl_hs_init_es
}
nlp_func_map.update(nlp_elastic_search_fmap)


def test_elastic_search():
    from pprint import pprint
    nlp_cfg = {
        "config_seq": ["import_funcs", "eapl_hs_init_elastic_setup", 'eapl_hs_init_es', "elastic_search_search",
                       "elastic_search_delete"],

        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "eapl_hs_init_setup": "from eapl_haystack_search import eapl_haystack_search_fmap"
            }
        },
        "eapl_hs_init_elastic_setup": {
            "func": "eapl_hs_init_setup",
            "hs_obj_key": "elastic_custom",
            "hs_setup_pipeline": [
                {
                    "func": "eapl_hs_docstore",
                    "hs_obj_key": "elastic_custom",
                    "docstore_type": "ElasticsearchDocumentStore",
                    "delete_all_docs_flag": False,
                    "docstore_params": {
                        "host": "13.232.238.41"
                    }
                }
            ]
        },
        'eapl_hs_init_es': {
            "func": "eapl_hs_init_es",
            'hs_obj_key': "elastic_custom"
        },
        "elastic_search_search": {
            "func": "eapl_es_op_cal",
            "api_cal": "search",
            "es_url_key": "testing",
            "es_out": "search",
            "api_cal_params": {
                'index': "cisco_data"
            }
        },
        "elastic_search_delete": {
            "func": "eapl_es_op_cal",
            "api_cal": "indices|delete",
            "es_url_key": "testing",
            "es_out": "delete",
            "api_cal_params": {
                "index": "test_index",
                "ignore": [400, 404]
            }
        },
    }

    data = {}

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)

    return None


if __name__ == '__main__':
    test_elastic_search()
