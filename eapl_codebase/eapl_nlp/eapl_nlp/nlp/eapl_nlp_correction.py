from urllib.request import urlopen
import wordninja
from spellchecker import SpellChecker
import logging

try:
    from .nlp_glob import *
    from .nlp_ops import *
except ImportError:
    from nlp_glob import *
    from nlp_ops import *

nlp_ee_logger = logging.getLogger(__name__)


def wordninja_load_model(data, cfg):
    f_path = cfg.get("f_path")
    data_key = cfg.get("data_key", "eapl_wordninja")
    nlp_ee_logger.debug("Starting wordninja Init")
    if data_key in nlp_glob.keys():
        wordninja_model = nlp_glob[data_key]
    else:
        if f_path:
            wordninja_zipfile = urlopen(f_path)
            wordninja_model = wordninja.LanguageModel(wordninja_zipfile)
        else:
            wordninja_model = wordninja
        nlp_glob[data_key] = wordninja_model

    data[data_key] = wordninja_model
    nlp_ee_logger.debug("Ending wordninja Init")
    return data


def wordninja_split(data, text_dict, op_dict):
    input_key = op_dict.get("input_key", "query")
    wordninja_key = op_dict.get("wordninja_key", "wordninja_key")
    output_key = op_dict.get("out_key", "tstring")
    wordninja_lm = data[wordninja_key]
    text_dict[output_key] = " ".join(wordninja_lm.split(text_dict[input_key]))

    return text_dict


def spell_checker_init(data, cfg):
    nlp_ee_logger.debug("Starting spell checker Init")
    data_key = cfg.get("data_key", "eapl_spellcheck")
    if data_key in nlp_glob.keys():
        spell_check = nlp_glob[data_key]
    else:
        spell_check = SpellChecker()
        nlp_glob[data_key] = spell_check

    data[data_key] = spell_check
    nlp_ee_logger.debug("Ending Spell Checker Init")
    return data


def spell_checker(data, text_dict, op_dict):
    input_key = op_dict.get("input_key", "query_doc")
    output_key = op_dict.get("out_key", "query_corrected")
    spell_check_key = op_dict.get("spell_check_key", "spell")
    query_doc = text_dict[input_key]
    corrected_lst = []
    for i in query_doc:
        if len(i.text) > 4:
            corrected_lst.append((data[spell_check_key].correction(i.text)))
        else:
            corrected_lst.append(i.text)
    query = " ".join(corrected_lst)
    text_dict[output_key] = query
    return text_dict


use_correction_map = {
    "spell_checker": spell_checker,
    "wordninja_split": wordninja_split,
    "wordninja_load_model": wordninja_load_model,
    "spell_checker_init": spell_checker_init
}

nlp_func_map.update(use_correction_map)


def test_use_pipeline():
    from pprint import pprint
    nlp_cfg = {
        "config_seq": ["init_pipe", "wordninja_load_model", "spell_checker_init", "record_nlp_ops", "manage_data_keys"],
        'init_pipe': {
            'func': 'eapl_nlp_pipeline_init',
        },
        "wordninja_load_model": {
            "func": "wordninja_load_model",
            "f_path": "https://s3.amazonaws.com/emplay.botresources/wordninja_files/word_nin.tar.gz",
            "out_key": "wordninja_lm_dct",
            "data_key": "eapl_wordninja"
        },
        "spell_checker_init": {
            "func": "spell_checker_init",
            "data_key": "eapl_spellcheck",
            "out_key": "spell"
        },
        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    "op": "create_spacy_doc",
                    "txt_key": "query",
                    "doc_key": "doc"
                },
                {
                    "op": "spell_checker",
                    'input_key': "doc",
                    'spell_check_key': "eapl_spellcheck",
                    'out_key': "query_corrected"
                },
                {
                    "op": "wordninja_split",
                    'input_key': 'query_corrected',
                    "wordninja_key": 'eapl_wordninja',
                    'out_key': "split_query"
                },
                {
                    "op": "manage_text_dict_keys",
                    "pop_keys": ['doc']
                },
            ]
        },

        'manage_data_keys': {
            'func': 'manage_data_keys',
            'pop_keys': ["nlp", "wordninja_lm_dct", "spell"]
        },
    }

    data_test = {
        "text_obj": [
            {
                "query": "s/4hana machinelearning desihn"

            },
            {
                "query": "cloud interfaace"

            },
        ]
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data_test = func(data_test, nlp_cfg)
    pprint(data_test['text_obj'])


if __name__ == '__main__':
    test_use_pipeline()
