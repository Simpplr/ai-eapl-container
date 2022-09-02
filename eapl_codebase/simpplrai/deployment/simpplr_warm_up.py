import json
import requests


def warm_up_call():
    url = "http://127.0.0.1:80/eapl/eapl-nlp/"

    payload = {
        "nlp_cfg": {
            "config_seq": [
                "import_funcs",
                "simpplr_config_api"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "simpplr_cfg_templates": "from .simpplr_cfg_templates import eapl_simpplr_cfg_temp_func_map"
                }
            },
            "simpplr_config_api": {
                "func": "simpplr_config_api",
                "config_name": "eapl_warm_up_simpplr"
            }
        },
        "data": {
            "substitutions": {
                "org_id": "test",
                "index": "test",
                "mongodb_database_name": "simpplranalytics",
                "mongodb_refresh": False
            },
            "text_obj": [
                {
                    "article_body": ""
                }
            ]
        }
    }
    headers = {
        'Authorization': 'Basic c2ltcHBscjpxd2VydHlAMTIz',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

    return response.text


if __name__ == '__main__':
    warm_up_call()
