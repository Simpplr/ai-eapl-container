import requests

nlp_func_map = dict()
nlp_glob = dict()

eapl_req_sess = requests.session()


def eapl_extract_nested_dict_value(data, nested_keys, delim="|", def_val=None):
    try:
        key_list = nested_keys.split(delim)
        val = data
        for key in key_list:
            val = val[key]
    except:
        val = def_val

    return val