from elasticsearch import Elasticsearch
from elasticsearch import helpers
from eapl_nlp.nlp.nlp_glob import *
import json


def init_es(data, cfg):
    es_obj = cfg.get('es_obj', 'es_obj')
    if es_obj not in nlp_glob.keys():
        es_params = cfg['es_params']
        scheme = es_params.get('scheme', 'http')
        host = es_params['host']
        port = es_params.get('port', 9200)
        username = es_params.get('username', '')
        password = es_params.get('password', '')
        es_host = f"{scheme}://{host}:{port}"
        es_client = Elasticsearch(es_host, http_auth=(username, password), timeout=30)
        nlp_glob[es_obj] = es_client
    else:
        es_client = nlp_glob[es_obj]

    data[es_obj] = es_client
    return data


def eapl_es_create_index(data, cfg):
    index = cfg['index']
    es_obj = cfg.get('es_obj', 'es_obj')
    es_client = data[es_obj]

    if not es_client.indices.exists(index=index):
        es_client.indices.create(index=index, ignore=400)

    return data


def eapl_es_add_records(data, cfg):
    index = cfg['index']
    es_obj = cfg.get('es_obj', 'es_obj')
    text_key = cfg.get('text_key', 'user_reco')
    indexing_method = cfg.get("indexing_method", "index_one_by_one")
    es_client = data[es_obj]

    if indexing_method == "index_bulk":
        helpers.bulk(es_client, data[text_key])
    else:
        for record in data[text_key]:
            es_client.index(index=index, body=record)

    return data


def refresh_index(data, cfg):
    index = cfg['index']
    es_obj = cfg.get('es_obj', 'es_obj')

    data[es_obj].indices.refresh(index=index)

    return data


def search_es(data, cfg):
    index = cfg['index']
    es_obj = cfg.get('es_obj', 'es_obj')
    size = cfg.get('size', '10')
    body = cfg.get("search_body", json.dumps({'query': {'match_all': {}}}))  # Handle default
    es_out = cfg.get("es_out", "es_res")

    try:
        es_res = data[es_obj].search(index=index, size=size, body=body)
        data[es_out] = es_res['hits']['hits']

    except:
        data[es_out] = []

    return data


def es_update(data, cfg):
    index = cfg['index']
    es_obj = cfg.get('es_obj', 'es_obj')
    body = cfg.get("update_body", json.dumps({'query': {'match_all': {}}}))
    es_update_key = cfg.get('es_update_key', 'es_index_update')

    try:
        data[es_update_key] = data[es_obj].update_by_query(index=index, body=body)
    except:
        data[es_update_key] = []

    return data


def es_delete_record(data, cfg):
    index = cfg['index']
    es_obj = cfg.get('es_obj', 'es_obj')
    body = cfg.get("del_body", json.dumps({'query': {'match_all': {}}}))
    es_delete_key = cfg.get('es_delete_key', 'es_index_delete')

    try:
        data[es_delete_key] = data[es_obj].delete_by_query(index=index, body=body)
    except:
        data[es_delete_key] = []

    return data


es_search_fmap = {
    "init_es": init_es,
    "refresh_index": refresh_index,
    "search_es": search_es,
    "es_update": es_update,
    "es_delete_record": es_delete_record,
    "eapl_es_create_index": eapl_es_create_index,
    "eapl_es_add_records": eapl_es_add_records
}
nlp_func_map.update(es_search_fmap)


def es_pipeline():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_es', 'refresh_index', 'es_delete_record'],
        'init_es': {
            'func': 'init_es',
        },
        'refresh_index': {
            'func': 'refresh_index',
            'index': "simplr_testing"
        },

        'es_update': {
            'func': 'es_update',
            'index': "simplr_testing",
            'update_body': {
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "Simpplr__Publish_End_DateTime__c": {
                                        "lt": "now"
                                    }
                                }
                            }
                        ]
                    }
                },
                "script": "ctx._source.expire_status = \"expired\";"
            }

        },

        'es_delete_record': {
            'func': 'es_delete_record',
            'index': "simplr_testing",
            'del_body': {
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "Simpplr__Publish_End_DateTime__c": {
                                        "lt": "now"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        },

        'search_es': {
            'func': 'search_es',
            'index': "simplr_testing",
            'size': 20,
            'search_body': {
                # "_source": {
                #     "includes": [
                #         "Simpplr__Publish_End_DateTime__c",
                #         "expire_status"
                #
                #     ]
                # },
                "query": {
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "Simpplr__Publish_End_DateTime__c": {
                                        "lt": "now"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        },

    }

    data = {}
    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)


if __name__ == '__main__':
    es_pipeline()
