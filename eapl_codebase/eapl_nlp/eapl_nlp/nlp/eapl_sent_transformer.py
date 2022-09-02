from sentence_transformers.cross_encoder import CrossEncoder
import logging
import pandas as pd
import Levenshtein

try:
    from .nlp_glob import nlp_func_map, nlp_glob
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import nlp_func_map, nlp_glob
    from nlp_ops import nlp_ops_funcs

t5_base_logger = logging.getLogger(__name__)


def eapl_cross_encoder_init(data, cfg):
    cr_obj = cfg.get('cr_obj', 'ce')
    model_name = cfg.get("model_name", "cross-encoder/stsb-distilroberta-base")
    ce_params = cfg.get("ce_params", {})
    model = nlp_glob.get(cr_obj, None)
    if model is None:
        model = CrossEncoder(model_name, **ce_params)
        nlp_glob[cr_obj] = model
    data[cr_obj] = model
    return data


def levi_distance(row, text1, text2):
    str1 = row[text1]
    str2 = row[text2]
    return Levenshtein.distance(str1, str2)


def levi_percentage(row, text1, text2):
    str1 = row[text1]
    str2 = row[text2]
    levi_score = row["levi_score"]
    max_len = max(len(str1), len(str2))
    return (max_len - levi_score) / max_len


def eapl_qg_evaluate(data, cfg):
    """
    This function takes list of records with 2 text keys as input , calculates Levenshtein distance and semantic score
    between them and adds as one more key.This also gives average of the scores to get an idea of what is the accuracy
    of the qg (across whole dataset).
    param :
        text_obj_key (str) : Data can have multiple text object keys , this tells which one to use it
        text1_key (str) : The key where the compare string is present
        text2_key (str) : The key where the compare string is present
        score_key (str) : This key tells which key name the score should store it as,
                          by default it uses 'score'
        avrg_key (str) : This key tells, what name the average of all score should be stored
        out_key (str) : When we add score, if we dont want the changes on the orginal data,
                        we can pass different output key, where the output will be stored.


        Eg : Input : {
                "text_obj": [
                    {
                        "input_id": 1002,
                        "question": "How to build your Matrix Grid reports in Succession?",
                        "manual_question": "How to build your Matrix Grid reports in Succession?"
                    },
                    {
                        "input_id": 1003,
                        "question": "How to configure 360 Templates using XML?",
                        "manual_question": "What is Post Review Phase while configuring 360 Templates Using XML?"
                    },
                    {
                        "input_id": 1004,
                        "question": "What are the technical API user in IAS?",
                        "manual_question": "What are the technical API user in Detailed Solution?"

                    }
                ],
            }
            Output : {
                "avrg_qg" : 0.7631473303326315,
                "text_obj": [
                    {
                        "input_id": 1002,
                        "question": "How to build your Matrix Grid reports in Succession?",
                        "manual_question": "How to build your Matrix Grid reports in Succession?",
                        "score" : 0.9952755570411682
                    },
                    {
                        "input_id": 1003,
                        "question": "How to configure 360 Templates using XML?",
                        "manual_question": "What is Post Review Phase while configuring 360 Templates Using XML?",
                        "score": 0.6649271109524895
                    },
                    {
                        "input_id": 1004,
                        "question": "What are the technical API user in IAS?",
                        "manual_question": "What are the technical API user in Detailed Solution?",
                        "score": 0.6292393230042368

                    }
                ],
            }
    """
    text_obj_key = cfg["text_obj_key"]
    cr_obj = cfg.get('cr_obj', 'ce')
    temp_data = {cr_obj: data[cr_obj], text_obj_key: data[text_obj_key]}
    text1_key = cfg["text1_key"]
    text2_key = cfg["text2_key"]
    score_key = cfg.get("score_key", "score")
    avrg_key = cfg.get("avrg_key", "avrg_qg")
    out_key = cfg.get("out_key", text_obj_key)
    df = pd.DataFrame(data[text_obj_key])
    temp_data = eapl_ce_similarity(data=temp_data, cfg=cfg)
    temp_df = pd.DataFrame(temp_data[out_key])
    temp_df["levi_score"] = temp_df.apply(levi_distance, axis=1, args=(text1_key, text2_key))
    temp_df["levi_percentage"] = temp_df.apply(levi_percentage, axis=1, args=(text1_key, text2_key))
    temp_df[score_key] = temp_df[["levi_percentage", score_key]].mean(axis=1)
    df[score_key] = temp_df[score_key]
    data[out_key] = df.to_dict("records")
    data[avrg_key] = temp_df[score_key].mean()
    return data


def eapl_ce_similarity(data, cfg):
    cr_obj = cfg.get('cr_obj', 'ce')
    text_obj_key = cfg["text_obj_key"]
    text1_key = cfg["text1_key"]
    text2_key = cfg["text2_key"]
    score_key = cfg.get("score_key", "score")
    out_key = cfg.get("out_key", text_obj_key)
    thres = cfg.get("thres", 0)

    cr_model = data[cr_obj]
    df = pd.DataFrame(data[text_obj_key])
    tmp_df = df.copy()
    input_lst = tmp_df[[text1_key, text2_key]].values.tolist()

    score_lst = cr_model.predict(input_lst)

    df[score_key] = score_lst
    df = df.query(f"{score_key} >= {thres}")
    data[out_key] = df.to_dict("records")

    return data


sent_trnsfrm_func_map = {
    'eapl_cross_encoder_init': eapl_cross_encoder_init,
    'eapl_ce_similarity': eapl_ce_similarity,
    'eapl_qg_evaluate': eapl_qg_evaluate
}

nlp_func_map.update(sent_trnsfrm_func_map)


def test_sent_transformer():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_cse', 'get_sim', 'manage_data_keys'],

        'init_cse': {
            'func': 'eapl_cross_encoder_init',
        },
        'get_sim': {
            'func': 'eapl_ce_similarity',
            'text_obj_key': 'text_obj',
            'thres': 0.80,
            'text1_key': 'text1_key',
            'text2_key': 'text2_key'

        },
        'manage_data_keys': {
            'func': 'manage_data_keys',
            'pop_keys': ['ce']
        }
    }

    data_test = {
        'text_obj': [
            {
                'input_id': 1002,
                'text1_key': "A man is eating food.",
                'text2_key': "A man is eating pasta."
            },
            {
                'input_id': 1003,
                'text1_key': "A man is eating food.",
                'text2_key': "A man went for walk."
            }
        ],

        "non_editable_fields": ["question_id"]
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data_test = func(data_test, nlp_cfg)
    pprint(data_test["text_obj"][:])


def test_qg_evaluate():
    from pprint import pprint
    nlp_cfg = {
        "config_seq": ["init_cse", "eapl_qg_evaluate", "manage_data_keys"],
        "init_cse": {
            "func": "eapl_cross_encoder_init",
        },
        "eapl_qg_evaluate": {
            "func": "eapl_qg_evaluate",
            "text_obj_key": "text_obj",
            "text1_key": "question",
            "text2_key": "manual_question",
            "out_key": "output"

        },
        "manage_data_keys": {
            "func": "manage_data_keys",
            "pop_keys": ["ce"]
        }
    }

    data_test = {
        "text_obj": [
            {
                "input_id": 1002,
                "question": "How to build your Matrix Grid reports in Succession?",
                "manual_question": "How to build your Matrix Grid reports in Succession?"
            },
            {
                "input_id": 1003,
                "question": "How to configure 360 Templates using XML?",
                "manual_question": "What is Post Review Phase while configuring 360 Templates Using XML?"
            },
            {
                "input_id": 1004,
                "question": "What are the technical API user in IAS?",
                "manual_question": "What are the technical API user in Detailed Solution?"

            }
        ],
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data_test = func(data_test, nlp_cfg)
    pprint(data_test)


if __name__ == '__main__':
    # test_sent_transformer()
    test_qg_evaluate()
