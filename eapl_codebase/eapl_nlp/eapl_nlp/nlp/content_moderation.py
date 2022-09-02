from profanityfilter import ProfanityFilter
import logging

try:
    from .nlp_glob import *
    from .nlp_ops import eapl_data_process_fk_flow
    from .nlp_utils import *

except ImportError:
    from nlp_glob import *
    from nlp_ops import eapl_data_process_fk_flow
    from nlp_utils import *

nlp_utils_logger = logging.getLogger(__name__)


def profanity_init(data, cfg):
    pf_key = cfg.get("pf_key", "profanityfilter")
    censor_char = cfg.get("censor_char", None)
    if pf_key not in nlp_glob.keys():
        pf = ProfanityFilter()
        nlp_glob[pf_key] = pf
    else:
        pf = nlp_glob[pf_key]

    if censor_char:
        pf.set_censor(censor_char)

    data[pf_key] = pf
    return data


def pf_extend_customlist(data, cfg):
    pf_key = cfg.get("pf_key", "profanityfilter")
    extend_list = cfg.get("extend_list")
    overwrite = cfg.get("overwrite", False)
    pf = data[pf_key]
    if overwrite:
        pf = ProfanityFilter(extra_censor_list=extend_list)
    else:
        custom_word_list = pf.get_extra_censor_list() + extend_list
        pf = ProfanityFilter(extra_censor_list=custom_word_list)

    nlp_glob[pf_key] = pf
    data[pf_key] = pf
    return data


def pf_remove_censorwords(data, cfg):
    pf_key = cfg.get("pf_key", "profanityfilter")
    remove_list = cfg.get("remove_list")
    pf = data[pf_key]
    for i in remove_list:
        pf.remove_word(i)
    nlp_glob[pf_key] = pf
    data[pf_key] = pf
    return data


def pf_define_customlist(data, cfg):
    pf_key = cfg.get("pf_key", "profanityfilter")
    defined_list = cfg.get("defined_list")
    overwrite = cfg.get("overwrite", False)
    pf = data[pf_key]
    if overwrite:
        pf.define_words(defined_list)
    else:
        custom_word_list = pf.get_custom_censor_list() + defined_list
        pf.define_words(custom_word_list)
    nlp_glob[pf_key] = pf
    data[pf_key] = pf
    return data


def profanity_filter(data, text_dict, op_dict):
    text_key = op_dict.get("text_key", "text")
    out_key = op_dict.get("out_key", "is_Profane")
    censor_out_key = op_dict.get("censor_out_key", "censored_text")
    pf_key = op_dict.get("pf_key", "profanityfilter")
    pf = data[pf_key]
    text = text_dict[text_key]
    text_dict[out_key] = pf.is_profane(text)
    text_dict[censor_out_key] = pf.censor(text)
    return text_dict


profanity_pproc_fmap = {
    "profanity_filter": profanity_filter,
    "pf_extend_customlist": pf_extend_customlist,
    "pf_define_customlist": pf_define_customlist,
    "profanity_init": profanity_init,
    "pf_remove_censorwords": pf_remove_censorwords

}

nlp_func_map.update(profanity_pproc_fmap)


def test_profanityfilter():
    from pprint import pprint
    data = {}
    nlp_cfg = {
        "config_seq": [
            "import_funcs",
            "profanity_init",
            "profanity_extend",
            "profanity_custom",
            "profanity_remove",
            "manage_data_keys"
        ],
        "import_custom_script": {
            "func": "eapl_download_import_file",
            "refresh": False,
            "file_info": [
                {
                    "file": "/home/chandana_upadya/Downloads/profanity_filter_custom.py",
                    "filemode": "fpath",
                    "import_stmt": "from profanity_filter_custom import profanity_pproc_fmap",
                    "func_key": "profanity_init"
                }
            ]
        },
        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "content_moderation": "from content_moderation import profanity_pproc_fmap"
            }
        },
        "profanity_init": {
            "func": "profanity_init",
            "pf_key": "profanityfilter"
        },
        "profanity_extend": {
            "func": "pf_extend_customlist",
            "overwrite": True,
            "extend_list": [
                "efgh"
            ]
        },
        "profanity_custom": {
            "func": "pf_define_customlist",
            "overwrite": True,
            "defined_list": [
                "here",
                "there"
            ]
        },
        "profanity_remove": {
            "func": "pf_remove_censorwords",
            "erase_word_list": True,
            "remove_list": [
                "shit"]

        },
        "manage_data_keys": {
            "func": "manage_data_keys",
            "pop_keys": [
                "nlp",
                "profanityfilter"
            ]
        }
    }
    pprint(data)
    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)


if __name__ == '__main__':
    test_profanityfilter()
