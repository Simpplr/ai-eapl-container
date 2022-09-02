from test.utils import *

topic_suggestion_test_cases = {
    "test_case_data": {
        "topic_suggestion_tc1": {
            "config_name": "topic_suggestion",
            "data": {
                "text_obj": [
                    {
                        "title": "What Is Data Science? A Beginner’s Guide To Data Science",
                        "text_intro": "As the world entered the era of big data, the need for its storage also grew. It was the main challenge and concern for the enterprise industries until 2010. The main focus was on building a framework and solutions to store data. Now when Hadoop and other frameworks have successfully solved the problem of storage, the focus has shifted to the processing of this data. Data Science is the secret sauce here. All the ideas which you see in Hollywood sci-fi movies can actually turn into reality by Data Science. Data Science is the future of Artificial Intelligence. Therefore, it is very important to understand what is Data Science and how can it add value to your business.",
                    }
                ]
            },
            "set_of_tags": {
                "data science",
                "big data",
                "main challenge",
                "enterprise industries",
                "secret sauce",
                "sci-fi movies",
                "artificial intelligence",
                "main focus",
                "data",
                "storage",
            },
            "set_of_fields": {
                "list_of_tags",
                "tags",
                "text_intro",
                "article_body",
                "title",
            },
            "status_code": 200,
        },
        "topic_suggestion_tc2": {
            "config_name": "topic_suggestion",
            "data": {
                "text_obj": [{"title": "<HTML> </HTML>", "text_intro": "<h1> </h1>"}]
            },
            "num_of_tags": 0,
            "status_code": 200,
        },
        "topic_suggestion_tc3": {
            "config_name": "topic_suggestion",
            "data": {"text_obj": [{"title": "123456789", "text_intro": "012458"}]},
            "num_of_tags": 0,
            "status_code": 200,
        },
        "topic_suggestion_tc4": {
            "config_name": "topic_suggestion",
            "data": {"text_obj": [{"title": "", "text_intro": "demo"}]},
            "error_type": "<class 'ValueError'>",
            "message": "HTTP Error 400: title null/nan/empty value",
            "status_code": 400,
        },
        "topic_suggestion_tc5": {
            "config_name": "topic_suggestion",
            "data": {"text_obj": [{"title": ".,/';][-=", "text_intro": "..,.,/.@#$"}]},
            "num_of_tags": 0,
            "status_code": 200,
        },
        "topic_suggestion_tc6": {
            "config_name": "topic_suggestion",
            "data": {
                "text_obj": [
                    {
                        "title": "Pollution is a term which even kids are aware of these days. It has become so common that almost everyone acknowledges the fact that pollution is rising continuously. The term ‘pollution’ means the manifestation of any unsolicited foreign substance in something. When we talk about pollution on earth, we refer to the contamination that is happening of the natural resources by various pollutants. All this is mainly caused by human activities which harm the environment in ways more than one. Therefore, an urgent need has arisen to tackle this issue straightaway. That is to say, pollution is damaging our earth severely and we need to realize its effects and prevent this damage. In this essay on pollution, we will see what are the effects of pollution and how to reduce it.",
                        "text_intro": "Pollution affects the quality of life more than one can imagine. It works in mysterious ways, sometimes which cannot be seen by the naked eye. However, it is very much present in the environment. For instance, you might not be able to see the natural gases present in the air, but they are still there. Similarly, the pollutants which are messing up the air and increasing the levels of carbon dioxide is very dangerous for humans. Increased level of carbon dioxide will lead to global warming.Further, the water is polluted in the name of industrial development, religious practices and more will cause a shortage of drinking water. Without water, human life is not possible. Moreover, the way waste is dumped on the land eventually ends up in the soil and turns toxic. If land pollution keeps on happening at this rate, we won’t have fertile soil to grow our crops on. Therefore, serious measures must be taken to reduce pollution to the core.After learning the harmful effects of pollution, one must get on the task of preventing or reducing pollution as soon as possible. To reduce air pollution, people should take public transport or carpool to reduce vehicular smoke. While it may be hard, avoiding firecrackers at festivals and celebrations can also cut down on air and noise pollution. Above all, we must adopt the habit of recycling. All the used plastic ends up in the oceans and land, which pollutes them.So, remember to not dispose of them off after use, rather reuse them as long as you can. We must also encourage everyone to plant more trees which will absorb the harmful gases and make the air cleaner. When talking on a bigger level, the government must limit the usage of fertilizers to maintain the soil’s fertility. In addition, industries must be banned from dumping their waste into oceans and rivers, causing water pollution.",
                    }
                ]
            },
            "max_num_of_tags": 10,
            "status_code": 200,
        },
    }
}


def test_tagging():
    topic_tc_dct = topic_suggestion_test_cases["test_case_data"]["topic_suggestion_tc1"]
    response = update_response(topic_tc_dct)

    # check for list of tags
    set_of_tags = set(response["text_obj"][0]["list_of_tags"])
    assert set_of_tags == topic_tc_dct["set_of_tags"]

    # check for fields in text_obj
    set_of_fields = set(response["text_obj"][0].keys())
    assert set_of_fields == topic_tc_dct["set_of_fields"]


def test_html_tags():
    topic_tc_dct = topic_suggestion_test_cases["test_case_data"]["topic_suggestion_tc2"]
    response = update_response(topic_tc_dct)

    # check for number of tags
    num_of_tags = len(response["text_obj"][0]["list_of_tags"])
    assert num_of_tags == topic_tc_dct["num_of_tags"]


def test_numerical_values():
    topic_tc_dct = topic_suggestion_test_cases["test_case_data"]["topic_suggestion_tc3"]
    response = update_response(topic_tc_dct)

    # check for number of tags
    num_of_tags = len(response["text_obj"][0]["list_of_tags"])
    assert num_of_tags == topic_tc_dct["num_of_tags"]


def test_empty_title():
    topic_tc_dct = topic_suggestion_test_cases["test_case_data"]["topic_suggestion_tc4"]
    response = update_response(topic_tc_dct)

    # check for error
    assert response["error_type"] == topic_tc_dct["error_type"]
    assert response["message"] == topic_tc_dct["message"]


def test_punctuations():
    topic_tc_dct = topic_suggestion_test_cases["test_case_data"]["topic_suggestion_tc5"]
    response = update_response(topic_tc_dct)

    # check for number of tags
    num_of_tags = len(response["text_obj"][0]["list_of_tags"])
    assert num_of_tags == topic_tc_dct["num_of_tags"]


def test_max_num_of_tags():
    topic_tc_dct = topic_suggestion_test_cases["test_case_data"]["topic_suggestion_tc6"]
    response = update_response(topic_tc_dct)

    # check for maximum number of tags
    num_of_tags = len(response["text_obj"][0]["list_of_tags"])
    assert num_of_tags == topic_tc_dct["max_num_of_tags"]
