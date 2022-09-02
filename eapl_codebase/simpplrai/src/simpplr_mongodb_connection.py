import pymongo
from pymongo import UpdateOne
import pandas as pd

from .nlp_glob import nlp_func_map, nlp_glob
from .nlp_ops import nlp_ops_funcs
from .nlp_utils import nlp_utils_fmap


def mongodb_connect(data, cfg):
    endpoint = cfg['endpoint'].copy()
    refresh = cfg.get('refresh', False)
    database_name = cfg.get('database_name', "simpplranalytics")
    output_key = cfg.get('output_key', "mongodb")
    mongodb_client = cfg.get('mongodb_client', 'mongodb_client')
    timeout_val = cfg.get("timeout_val", 10000)
    endpoint.update({'serverSelectionTimeoutMS': timeout_val})
    test_conn = cfg.get('test_conn', False)
    if output_key not in nlp_glob or refresh:
        client = pymongo.MongoClient(**endpoint)
        client.server_info()
        dbnames = client.list_database_names()
        if database_name in dbnames:
            db = client[database_name]
        else:
            raise ValueError(f"HTTP Error 400: db {database_name} not found")
        nlp_glob[mongodb_client] = client
        nlp_glob[output_key] = db

    if test_conn:
        client = nlp_glob[mongodb_client]
        client.server_info()

    return data


def mongodb_load_table(data, cfg):
    mongo_db_key = cfg.get('mongo_db_key', 'mongodb')
    collection_name = cfg['collection_name']
    output_key = cfg['output_key']
    query = cfg.get('query', {})
    projection = cfg.get('projection', None)
    out_format = cfg.get('out_format', 'json')

    db = nlp_glob[mongo_db_key]
    col = db[collection_name]
    records = list(col.find(query, projection))

    if out_format == 'df':
        output = pd.DataFrame.from_records(records)
        if projection:
            cols = list(projection.keys())
            output = output.reindex(columns=cols)
            # Required to handle empty dataframes. Default data type is float
            output = output.astype(str)
    else:
        output = records

    data[output_key] = output
    return data


def mongodb_load_mul_table(data, cfg):
    tables_info = cfg.get('tables_info', [])

    for table_details in tables_info:
        cfg_table = cfg.copy()
        cfg_table.update(table_details)
        data = mongodb_load_table(data, cfg_table)

    return data


def mongodb_write(data, cfg):
    mongo_db_key = cfg.get('mongo_db_key', 'mongodb')
    collection_name = cfg['collection_name']
    input_data_key = 'write_input_data'
    extra_params = cfg.get('extra_params', {})
    db = nlp_glob[mongo_db_key]
    col = db[collection_name]
    input_data = data[input_data_key]
    col.bulk_write(input_data, **extra_params)
    return data


eapl_simpplr_mogodb_fmap = {
    'mongodb_connect': mongodb_connect,
    'mongodb_load_table': mongodb_load_table,
    'mongodb_load_mul_table': mongodb_load_mul_table,
    'mongodb_write': mongodb_write
}

nlp_func_map.update(eapl_simpplr_mogodb_fmap)


def test_mdb_funcs():
    data = {}
    data = mongodb_connect(data, cfg={'endpoint': {"host": "localhost", "port": 27017, "maxPoolSize": 50},
                                      'database_name': "simpplranalytics",
                                      'output_key': "mongodb"})
    data = mongodb_load_table(data, cfg={'mongo_db_key': 'mongodb',
                                         'collection_name': "simpplr_content_datas",
                                         'output_key': 'user_content_interaction',
                                         'query': {'org_id': 'org_1'}})

    print(data)


if __name__ == '__main__':
    test_mdb_funcs()
