from test.utils import update_response
from test.conftest import SECRET
import copy


ORG_ID = "00Do0000000bssxEAA"
MDB_NAME = "simpplranalytics"
INDEX = ORG_ID.lower()
VERSION = SECRET.version
RECO_USER_IDS = ["a2L1N000001ZwlGUAS", "a2L1N000001RAtJUAW"]
RECO_EMAIL_USER = "a2L1N000001h8EwUAI"
TOPN = 10

data = {
    "setup": {
        "substitutions": {
            "org_id": f"{ORG_ID}",
            "index": f"{INDEX}",
            "mongodb_database_name": f"{MDB_NAME}",
            "mongodb_refresh": True,
        },
        "test_freeze_time1": "2021-11-01",
    },
    "realtime": {
        "substitutions": {
            "version": f"{VERSION}",
            "org_id": f"{ORG_ID}",
            "index": f"{INDEX}",
            "topn": TOPN,
            "mongodb_database_name": f"{MDB_NAME}",
            "mongodb_refresh": True,
            "reco_user_ids": f"{RECO_USER_IDS}",
        },
        "source": "feed"
    },
}

errors = {}

page_recomendation_test_cases = {
    "test_case_data": {
        "pr_setup_check_version": {
            "config_name": "page_reco_setup",
            "data": None,
            "status_code": 200,
        },
        "pr_setup_check_dbname": {
            "config_name": "page_reco_setup",
            "data": None,
            "status_code": 400,
        },
        "pr_setup_check_empty_org_id": {
            "config_name": "page_reco_setup",
            "data": None,
            "status_code": 400,
        },
        "pr_setup_check_empty_index": {
            "config_name": "page_reco_setup",
            "data": None,
            "status_code": 400,
        },
        "pr_rt_check_keys": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 200
        },
        "pr_rt_check_for_1_user": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 200,
        },
        "pr_rt_check_for_multi_user": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 200,
        },
        "pr_rt_check_for_cs_user": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 200,
        },
        "pr_rt_check_topn": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 400,
        },
        "pr_rt_check_dbname": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 400,
        },
        "pr_rt_check_empty_userid": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 400,
        },
        "pr_rt_check_email": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 200
        },
        "pr_rt_check_mustread": {
            "config_name": "page_reco_realtime",
            "data": None,
            "status_code": 200
        }
    }
}


def test_pr_setup_check_version():
    _data = data["setup"].copy()
    CONFIG = page_recomendation_test_cases["test_case_data"]["pr_setup_check_version"]
    CONFIG["data"] = _data

    response = update_response(CONFIG)

    assert response["version"] == SECRET.version


def test_pr_setup_check_dbname():
    MDB_NAME = "simpplran"

    _data = data["setup"].copy()
    _data["substitutions"]["mongodb_database_name"] = MDB_NAME
    _data["substitutions"]["mongodb_refresh"] = True

    CONFIG = page_recomendation_test_cases["test_case_data"]["pr_setup_check_dbname"]
    CONFIG["data"] = _data

    response = update_response(CONFIG)

    assert response["status"] == "fail"
    assert response["message"] == f"HTTP Error 400: db {MDB_NAME} not found"
    assert response["error_type"] == "<class 'ValueError'>"


def test_pr_setup_check_empty_org_id():
    _data = data["setup"].copy()
    _data["substitutions"]["org_id"] = ""
    CONFIG = page_recomendation_test_cases["test_case_data"][
        "pr_setup_check_empty_org_id"
    ]
    CONFIG["data"] = _data

    response = update_response(CONFIG)

    assert response["status"] == "fail"
    assert response["message"] == "HTTP Error 400: org_id null/nan/empty value" 


def test_pr_realtime_check_keys():
    _data = data["realtime"].copy()
    _data["substitutions"]["reco_user_ids"] = RECO_USER_IDS[0]
    
    CONFIG = page_recomendation_test_cases["test_case_data"]["pr_rt_check_keys"]
    CONFIG["data"] = _data

    response = update_response(CONFIG)

    assert response["status"] == "success"
    assert any( x in ["Simpplr__Content__Id", "score", "message", "reco_method"] for x in list(response["content_recommendations"][RECO_USER_IDS[0]][0].keys()))


def test_pr_realtime_check_1_user():
    _data = data["realtime"].copy()
    _data["substitutions"]["reco_user_ids"] = RECO_USER_IDS[0]

    CONFIG = page_recomendation_test_cases["test_case_data"]["pr_rt_check_for_1_user"]
    CONFIG["data"] = _data

    response = update_response(CONFIG)

    assert RECO_USER_IDS[0] in response["content_recommendations"].keys()
    assert 1 <= len(response["content_recommendations"][RECO_USER_IDS[0]]) <= 10


def test_pr_realtime_check_multi_user():
    _data = data["realtime"].copy()
    _data["substitutions"]["reco_user_ids"] = RECO_USER_IDS
    CONFIG = page_recomendation_test_cases["test_case_data"][
        "pr_rt_check_for_multi_user"
    ]
    CONFIG["data"] = _data

    response = update_response(CONFIG)

    assert RECO_USER_IDS == list(response["content_recommendations"].keys())
    for id in response["content_recommendations"]:
        assert 1 <= len(response["content_recommendations"][id]) <= 10


def test_pr_realtime_check_topn():
    _data = copy.deepcopy(data["realtime"])
    _data["substitutions"]["topn"] = 0
    _data["substitutions"]["reco_user_ids"] = RECO_USER_IDS[0]

    CONFIG = page_recomendation_test_cases["test_case_data"]["pr_rt_check_topn"]
    CONFIG["data"] = _data

    response = update_response(CONFIG)

    assert response["status"] == "fail"
    assert response["message"] == "HTTP Error 400: topn value not in range"


def test_pr_rt_check_email():
    _data = data["realtime"].copy()
    _data["substitutions"]["reco_user_ids"] = RECO_EMAIL_USER
    _data["source"] = "email"

    CONFIG = page_recomendation_test_cases["test_case_data"]["pr_rt_check_email"]
    CONFIG["data"] = _data

    res1 = update_response(CONFIG)
    res2 = update_response(CONFIG)
    
    assert res1 == res2
    
    res3 = update_response(CONFIG)
    
    assert res3 != res1 and res3 != res2
    

def test_pr_rt_check_mustread():
    _data = data["realtime"].copy()
    _data["substitutions"]["reco_user_ids"] = RECO_USER_IDS
    _data["source"] = "feed"

    CONFIG = page_recomendation_test_cases["test_case_data"]["pr_rt_check_mustread"]
    CONFIG["data"] = _data
    
    response = update_response(CONFIG)
    
    for _ in range(3):
        res = update_response(CONFIG)
        assert res == response