import json, requests
from requests.auth import HTTPBasicAuth
from test.conftest import SECRET

generic_info = {
    "nlp_cfg": {
        "config_seq": ["import_funcs", "simpplr_config_api"],
        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "simpplr_cfg_templates": "from .simpplr_cfg_templates import eapl_simpplr_cfg_temp_func_map"
            },
        },
        "simpplr_config_api": {"func": "simpplr_config_api"},
    }
}


def update_config_name(config_name):
    nlp_cfg = generic_info["nlp_cfg"].copy()
    nlp_cfg["simpplr_config_api"]["config_name"] = config_name
    return nlp_cfg


def update_payload(config_name, data):
    payload = dict(nlp_cfg=update_config_name(config_name), data=data)
    return payload


def update_response(tc_dct):
    payload = update_payload(tc_dct["config_name"], tc_dct["data"])
    payload = json.dumps(payload)
    response = requests.post(
        SECRET.url, data=payload, auth=HTTPBasicAuth(SECRET.username, SECRET.password)
    )

    # check for request status
    assert response.status_code == tc_dct["status_code"]

    response = response.json()

    if tc_dct["status_code"] == 200:
        # check for version
        assert response["version"] == SECRET.version

    return response
