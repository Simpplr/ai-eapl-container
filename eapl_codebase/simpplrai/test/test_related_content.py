from test.utils import *

related_content_test_cases = {
    "test_case_data": {
        "indexing_tc1": {
            "config_name": "post_data_redis",
            "data": {
                "substitutions": {"org_id": "testing", "index": "testing"},
                "text_obj": [
                    {
                        "type": "Page",
                        "topic_list": None,
                        "title": "Another test",
                        "text_intro": "<p>test</p>",
                        "site_type": "Public",
                        "site_id": "a1D5Y000005RN30UAG",
                        "page_category": "Test Page",
                        "id": "a1B5Y00000Gd0PZUAOk",
                        "expires_at": "",
                        "publishStartDate": "2017-07-23T18:30:00.000+0000",
                    },
                    {
                        "type": "Event",
                        "topic_list": None,
                        "title": "Test Event",
                        "text_intro": "<p>test event comment</p>",
                        "site_type": "Public",
                        "site_id": "a1D5Y000005RN30UAG",
                        "page_category": None,
                        "id": "a1B5Y00000GdEbXUAW",
                        "expires_at": "2019-07-23T18:30:00.000+0000",
                        "publishStartDate": "2017-06-23T18:30:00.000+0000",
                    },
                ],
            },
            "ref_response": {"org_id": "testing", "index": "testing"},
            "status_code": 200,
        },
        "indexing_tc2": {
            "config_name": "post_data_redis",
            "data": {
                "substitutions": {"org_id": "testing", "index": "testing"},
                "text_obj": [
                    {
                        "type": "Page",
                        "topic_list": None,
                        "title": "Another test",
                        "text_intro": "<p>test</p>",
                        "site_type": "Public",
                        "site_id": "a1D5Y000005RN30UAG",
                        "page_category": "Test Page",
                        "id": "",
                        "expires_at": "",
                        "publishStartDate": "2017-07-23T18:30:00.000+0000",
                    }
                ],
            },
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: id null/nan/empty value",
            "status_code": 400,
        },
        "indexing_tc3": {
            "config_name": "post_data_redis",
            "data": {
                "substitutions": {"org_id": "testing", "index": "testing"},
                "text_obj": [
                    {
                        "type": "Page",
                        "topic_list": None,
                        "title": "",
                        "text_intro": "<p>test</p>",
                        "site_type": "Public",
                        "site_id": "a1D5Y000005RN30UAG",
                        "page_category": "Test Page",
                        "id": "a1B5Y00000Gd0PZUAOk",
                        "expires_at": "",
                        "publishStartDate": "2017-07-23T18:30:00.000+0000",
                    }
                ],
            },
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: title null/nan/empty value",
            "status_code": 400,
        },
        "indexing_tc4": {
            "config_name": "post_data_redis",
            "data": {
                "substitutions": {"org_id": "testing", "index": "testing"},
                "text_obj": [
                    {
                        "type": "Page",
                        "topic_list": None,
                        "title": "Another test",
                        "text_intro": "<p>test</p>",
                        "site_type": "",
                        "site_id": "a1D5Y000005RN30UAG",
                        "page_category": "Test Page",
                        "id": "a1B5Y00000Gd0PZUAOk",
                        "expires_at": "",
                        "publishStartDate": "2017-07-23T18:30:00.000+0000",
                    }
                ],
            },
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: site_type null/nan/empty value",
            "status_code": 400,
        },
        "indexing_tc5": {
            "config_name": "post_data_redis",
            "data": {
                "substitutions": {"org_id": "testing2", "index": "testing"},
                "text_obj": [
                    {
                        "type": "Page",
                        "topic_list": None,
                        "title": "Another test",
                        "text_intro": "<p>test</p>",
                        "site_type": "Public",
                        "site_id": "a1D5Y000005RN30UAG",
                        "page_category": "Test Page",
                        "id": "a1B5Y00000Gd0PZUAOk",
                        "expires_at": "",
                        "publishStartDate": "2017-07-23T18:30:00.000+0000",
                    }
                ],
            },
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: ('org_id', 'index') not in sync",
            "status_code": 400,
        },
        "setup_tc1": {
            "config_name": "recommendations_setup",
            "data": {"substitutions": {"org_id": "testing", "index": "testing"}},
            "status_code": 200,
        },
        "setup_tc2": {
            "config_name": "recommendations_setup",
            "data": {"substitutions": {"org_id": "testingabc", "index": "testingabc"}},
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: index not in redis/connection failed",
            "status_code": 400,
        },
        "setup_tc3": {
            "config_name": "recommendations_setup",
            "data": {"substitutions": {"org_id": "testingabc", "index": "testing"}},
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: ('org_id', 'index') not in sync",
            "status_code": 400,
        },
        "real_time_recommendations_tc1": {
            "config_name": "real_time_reco",
            "data": {
                "substitutions": {
                    "org_id": "testing",
                    "index": "testing",
                    "id": "a1B5Y00000Gd0PZUAOk",
                }
            },
            "fields": [
                "text",
                "score",
                "probability",
                "question",
                "id",
                "expires_at",
                "text_intro",
                "site_type",
                "publishStartDate",
                "site_id",
                "page_category",
                "type",
                "topic_list",
                "title",
                "doc_id",
                "similarity_score",
                "recency_rate",
            ],
            "recommendation_count": 2,
            "status_code": 200,
        },
        "real_time_recommendations_tc2": {
            "config_name": "real_time_reco",
            "data": {
                "substitutions": {
                    "org_id": "testingabc",
                    "index": "testingabc",
                    "id": "a1B5Y00000Gd0PZUAOk",
                }
            },
            "error_type": "<class 'elasticsearch.exceptions.NotFoundError'>",
            "message": "NotFoundError(404, 'index_not_found_exception', 'no such index [rc_testingabc]', rc_testingabc, index_or_alias)",
            "status_code": 404,
        },
        "real_time_recommendations_tc3": {
            "config_name": "real_time_reco",
            "data": {
                "substitutions": {
                    "org_id": "testing",
                    "index": "testing",
                    "id": "a1B5Y00000Gd0PZUAOk2",
                }
            },
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: Embedding for given id not found",
            "status_code": 400,
        },
        "real_time_recommendations_tc4": {
            "config_name": "real_time_reco",
            "data": {
                "substitutions": {
                    "org_id": "testing2",
                    "index": "testing",
                    "id": "a1B5Y00000Gd0PZUAOk",
                }
            },
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: ('org_id', 'index') not in sync",
            "status_code": 400,
        },
        "unpublished_article_tc1": {
            "config_name": "remove_unpublished_content",
            "data": {
                "substitutions": {
                    "org_id": "testing",
                    "index": "testing",
                    "id": "a1B5Y00000Gd0PZUAOk",
                }
            },
            "deleted_count": 1,
            "status_code": 200,
        },
        "unpublished_article_tc2": {
            "config_name": "remove_unpublished_content",
            "data": {
                "substitutions": {
                    "org_id": "testingabc",
                    "index": "testingabc",
                    "id": "a1B5Y00000Gd0PZUAOk",
                }
            },
            "error_type": "<class 'elasticsearch.exceptions.NotFoundError'>",
            "message": "NotFoundError(404, 'index_not_found_exception', 'no such index [rc_testingabc]', rc_testingabc, index_or_alias)",
            "status_code": 404,
        },
        "unpublished_article_tc3": {
            "config_name": "remove_unpublished_content",
            "data": {
                "substitutions": {
                    "org_id": "testing2",
                    "index": "testing",
                    "id": "a1B5Y00000Gd0PZUAOk",
                }
            },
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: ('org_id', 'index') not in sync",
            "status_code": 400,
        },
        "expired_content_tc1": {
            "config_name": "remove_expired_content",
            "data": {"substitutions": {"org_id": "testing", "index": "testing"}},
            "deleted_count": 1,
            "status_code": 200,
        },
        "expired_content_tc2": {
            "config_name": "remove_expired_content",
            "data": {"substitutions": {"org_id": "testingabc", "index": "testingabc"}},
            "error_type": "<class 'elasticsearch.exceptions.NotFoundError'>",
            "message": "NotFoundError(404, 'index_not_found_exception', 'no such index [rc_testingabc]', rc_testingabc, index_or_alias)",
            "status_code": 404,
        },
        "expired_content_tc3": {
            "config_name": "remove_expired_content",
            "data": {"substitutions": {"org_id": "testing2", "index": "testing"}},
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: ('org_id', 'index') not in sync",
            "status_code": 400,
        },
    }
}


def test_indexing():
    indexing_tc_dct = related_content_test_cases["test_case_data"]["indexing_tc1"]
    response = update_response(indexing_tc_dct)

    # check for index and org_id
    response = response["substitutions"]
    assert response == indexing_tc_dct["ref_response"]


def test_indexing_empty_id():
    indexing_tc_dct = related_content_test_cases["test_case_data"]["indexing_tc2"]
    response = update_response(indexing_tc_dct)

    # check for error
    assert response["error_type"] == indexing_tc_dct["error_type"]
    assert response["message"] == indexing_tc_dct["message"]


def test_indexing_empty_title():
    indexing_tc_dct = related_content_test_cases["test_case_data"]["indexing_tc3"]
    response = update_response(indexing_tc_dct)

    # check for error
    assert response["error_type"] == indexing_tc_dct["error_type"]
    assert response["message"] == indexing_tc_dct["message"]


def test_indexing_empty_site_type():
    indexing_tc_dct = related_content_test_cases["test_case_data"]["indexing_tc4"]
    response = update_response(indexing_tc_dct)

    # check for error
    assert response["error_type"] == indexing_tc_dct["error_type"]
    assert response["message"] == indexing_tc_dct["message"]


def test_indexing_mismatch_id_org_id():
    indexing_tc_dct = related_content_test_cases["test_case_data"]["indexing_tc5"]
    response = update_response(indexing_tc_dct)

    # check for error
    assert response["error_type"] == indexing_tc_dct["error_type"]
    assert response["message"] == indexing_tc_dct["message"]


def test_setup():
    setup_tc_dct = related_content_test_cases["test_case_data"]["setup_tc1"]
    response = update_response(setup_tc_dct)


def test_setup_wrong_index():
    setup_tc_dct = related_content_test_cases["test_case_data"]["setup_tc2"]
    response = update_response(setup_tc_dct)

    # check for error
    assert response["error_type"] == setup_tc_dct["error_type"]
    assert response["message"] == setup_tc_dct["message"]


def test_setup_mismatch_id_org_id():
    setup_tc_dct = related_content_test_cases["test_case_data"]["setup_tc3"]
    response = update_response(setup_tc_dct)

    # check for error
    assert response["error_type"] == setup_tc_dct["error_type"]
    assert response["message"] == setup_tc_dct["message"]


def test_recommendations():
    recommendation_tc_dct = related_content_test_cases["test_case_data"][
        "real_time_recommendations_tc1"
    ]
    response = update_response(recommendation_tc_dct)

    fields = list(response["similar_text"][0][0].keys())
    count = len(response["similar_text"][0])

    # check for recommendation_fields
    assert fields == recommendation_tc_dct["fields"]

    # check for recommendations_count
    assert count == recommendation_tc_dct["recommendation_count"]


def test_recommendations_wrong_index():
    recommendation_tc_dct = related_content_test_cases["test_case_data"][
        "real_time_recommendations_tc2"
    ]
    response = update_response(recommendation_tc_dct)

    # check for error
    assert response["error_type"] == recommendation_tc_dct["error_type"]
    assert response["message"] == recommendation_tc_dct["message"]


def test_recommendations_wrong_id():
    recommendation_tc_dct = related_content_test_cases["test_case_data"][
        "real_time_recommendations_tc3"
    ]
    response = update_response(recommendation_tc_dct)

    # check for error
    assert response["error_type"] == recommendation_tc_dct["error_type"]
    assert response["message"] == recommendation_tc_dct["message"]


def test_recommendations_mismatch_id_org_id():
    recommendation_tc_dct = related_content_test_cases["test_case_data"][
        "real_time_recommendations_tc4"
    ]
    response = update_response(recommendation_tc_dct)

    # check for error
    assert response["error_type"] == recommendation_tc_dct["error_type"]
    assert response["message"] == recommendation_tc_dct["message"]


def test_unpublished_article():
    unpublished_tc_dct = related_content_test_cases["test_case_data"][
        "unpublished_article_tc1"
    ]
    response = update_response(unpublished_tc_dct)

    # check for number of deleted document
    response = response["es_index_delete"]["deleted"]
    assert response == unpublished_tc_dct["deleted_count"]


def test_unpublished_article_wrong_index():
    unpublished_tc_dct = related_content_test_cases["test_case_data"][
        "unpublished_article_tc2"
    ]
    response = update_response(unpublished_tc_dct)

    # check for error
    assert response["error_type"] == unpublished_tc_dct["error_type"]
    assert response["message"] == unpublished_tc_dct["message"]


def test_unpublished_article_mismatch_id_org_id():
    unpublished_tc_dct = related_content_test_cases["test_case_data"][
        "unpublished_article_tc3"
    ]
    response = update_response(unpublished_tc_dct)

    # check for error
    assert response["error_type"] == unpublished_tc_dct["error_type"]
    assert response["message"] == unpublished_tc_dct["message"]


def test_expired_content():
    expired_tc_dct = related_content_test_cases["test_case_data"]["expired_content_tc1"]
    response = update_response(expired_tc_dct)

    # check for number of deleted document
    response = response["es_index_delete"]["deleted"]
    assert response == expired_tc_dct["deleted_count"]


def test_expired_content_wrong_index():
    expired_tc_dct = related_content_test_cases["test_case_data"]["expired_content_tc2"]
    response = update_response(expired_tc_dct)

    # check for error
    assert response["error_type"] == expired_tc_dct["error_type"]
    assert response["message"] == expired_tc_dct["message"]


def test_expired_content_mismatch_id_org_id():
    expired_tc_dct = related_content_test_cases["test_case_data"]["expired_content_tc3"]
    response = update_response(expired_tc_dct)

    # check for error
    assert response["error_type"] == expired_tc_dct["error_type"]
    assert response["message"] == expired_tc_dct["message"]
