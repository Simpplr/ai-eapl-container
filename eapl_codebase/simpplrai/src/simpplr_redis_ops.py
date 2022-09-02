import redis
import logging
import json

from .nlp_glob import *
from .nlp_ops import *
from .eapl_haystack_search import eapl_hs_get_all_docs

nlp_tags_logger = logging.getLogger(__name__)


def save_data_redis(data, cfg):
    redis_conn = nlp_glob.get('redis', None)
    c_name = cfg.get("org_id")
    input_key = cfg.get('input_key', 'text_obj')
    id_col = cfg.get('id_col', 'Id')

    if redis_conn:
        if redis_conn.exists(c_name):
            redis_data = redis_conn.get(c_name)
            redis_data = json.loads(redis_data)

            for rec in data[input_key]:
                my_item = next((item for item in redis_data[input_key] if item[id_col] == rec[id_col]), None)
                if my_item:
                    redis_data[input_key].remove(my_item)

            if input_key in redis_data.keys():
                redis_data[input_key] = redis_data[input_key] + data[input_key]
                redis_conn.mset({c_name: json.dumps(redis_data)})
            else:
                redis_conn.mset({c_name: json.dumps({input_key: data[input_key]})})
        else:
            redis_conn.mset({c_name: json.dumps({input_key: data[input_key]})})

    return data


def start_redis(data, cfg):
    redis_init_params = cfg.get("redis_init_params", {})
    redis_name = cfg.get("redis_name", "redis")
    refresh = cfg.get('refresh', False)
    test_conn = cfg.get("test_conn", False)

    if redis_name not in nlp_glob or refresh:
        try:
            if 'url' in redis_init_params:
                redis_conn = redis.from_url(**redis_init_params)
            else:
                redis_conn = redis.StrictRedis(**redis_init_params)
            redis_conn.ping()
        except Exception as e:
            redis_conn = None
            nlp_tags_logger.info(f"Redis not initialised..", e)
            raise ValueError(f"HTTP Error 400: Redis connection issue")

        nlp_glob[redis_name] = redis_conn

    if test_conn:
        redis_conn = nlp_glob[redis_name]
        redis_conn.ping()

    return data


def read_data_redis(data, cfg):
    redis_conn = nlp_glob.get('redis', None)
    c_name = cfg.get("org_id")
    input_key = cfg.get('input_key', 'text_obj')

    if redis_conn.exists(c_name):
        redis_data = redis_conn.get(c_name)
        redis_data = json.loads(redis_data)

        if input_key in redis_data.keys():
            data[input_key] = []
            for val in redis_data[input_key]:
                data[input_key].append(val)
        else:
            nlp_tags_logger.info("no data found while loading data to redis")
            pass
    else:
        nlp_tags_logger.info("redis connection failed")
        raise ValueError(f"HTTP Error 400: index not in redis/connection failed")
        pass

    return data


def delete_data_redis(data, cfg):
    redis_conn = nlp_glob.get('redis', None)
    c_name = cfg.get("org_id")
    input_key = cfg.get('input_key', 'text_obj')
    id_col = cfg.get('id_col', 'id')
    id_val_key = cfg.get('id_val_key', None)
    id_val = cfg.get('id_val', [])

    if id_val_key:
        id_val = data[id_val_key]

    if redis_conn:
        if redis_conn.exists(c_name):
            redis_data = redis_conn.get(c_name)
            redis_data = json.loads(redis_data)

            all_ids = [sub[id_col] for sub in redis_data[input_key]]

            if type(id_val) == list:
                redis_data[input_key] = [d for d in redis_data[input_key] if d[id_col] not in all_ids]

            else:
                for item in redis_data[input_key]:
                    if item[id_col] == id_val:
                        redis_data[input_key].remove(item)

            if input_key in redis_data.keys():
                redis_conn.mset({c_name: json.dumps(redis_data)})
            else:
                pass
        else:
            pass

    return data


def redis_get_ids(data, cfg):
    redis_conn = nlp_glob.get('redis', None)
    c_name = cfg.get("org_id")
    input_key = cfg.get('input_key', 'text_obj')
    id_col = cfg.get('id_col', 'id')
    out_key = cfg.get('out_key', 'redis_ids')
    content_ids = []

    if redis_conn:
        if redis_conn.exists(c_name):
            redis_data = redis_conn.get(c_name)
            redis_data = json.loads(redis_data)

            content_ids = [sub[id_col] for sub in redis_data[input_key]]
        else:
            pass

    data[out_key] = content_ids

    return data


def rc_flush_redis_ids(data, cfg):
    meta_key_name = cfg.get('meta_key_name', 'doc_id')
    id_val_key = cfg.get('id_val_key', 'doc_id_val')
    data_ids_key = cfg.get('data_ids_key', 'redis_ids')
    get_docs_params = cfg.get('get_docs_params', {})
    out_key = cfg.get('out_key', 'docs_gad')
    unique_ids = data[data_ids_key]

    get_docs_params.update({"ids": unique_ids})
    content_data = eapl_hs_get_all_docs(data, cfg)

    doc_meta_values = [sub.__dict__['meta'][meta_key_name] for sub in content_data[out_key]]
    data[id_val_key] = doc_meta_values
    delete_data_redis(data, cfg)
    del data[id_val_key]

    return data


simpplr_redis_ops_fmap = {
    "save_data_redis": save_data_redis,
    "start_redis": start_redis,
    "read_data_redis": read_data_redis,
    "delete_data_redis": delete_data_redis,
    "rc_flush_redis_ids": rc_flush_redis_ids,
    "redis_get_ids": redis_get_ids,
}
nlp_func_map.update(simpplr_redis_ops_fmap)
