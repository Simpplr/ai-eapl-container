import re

from .nlp_glob import *
from .simpplr_configs import *

nlp_p_logger = logging.getLogger(__name__)


# TODO: Temp setup for backward compatibility
def map_config_name(config_name):
    return re.sub("_v\\d+.?\\d*$", '', config_name)


def simpplr_config_api(data, cfg):
    config_name = cfg.get("config_name")
    config_name = map_config_name(config_name)
    version = eapl_extract_nested_dict_value(data, "substitutions|version", delim="|", def_val="1.0")
    func = nlp_func_map['eapl_data_process_fk_flow']
    nlp_p_logger.info(f"Code version: {simpplrai_code_version}")
    nlp_p_logger.debug(f"simpplr_nlp_config: {simpplr_config[config_name]}")
    data = func(data, simpplr_config[config_name])
    data.update({"version": version})
    if config_name not in ['simpplr_healthcheck']:
        data.update({"code_version": simpplrai_code_version})
    nlp_p_logger.info(f"Code version: {simpplrai_code_version}")
    return data


eapl_simpplr_cfg_temp_func_map = {
    'simpplr_config_api': simpplr_config_api
}
nlp_func_map.update(eapl_simpplr_cfg_temp_func_map)


# sample test body
def test_use_pipeline():
    from pprint import pprint
    nlp_cfg = {
        "config_seq": [
            "import_funcs",
            "call_config"
        ],
        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "simpplr_cfg_templates": "from .simpplr_cfg_templates import eapl_simpplr_cfg_temp_func_map"
            }
        },
        "call_config": {
            "func": "call_config",
            "config_name": "simpplr_topic_rec"
        }
    }
    data = {
        "substitutions": {
            "rake_score": "score > 2",
            "rake_topn": "10",
            "semantic_drop_threshold": "0.5"
        },
        "text_obj": [
            {
                "title": "testing topic suggestion ",
                "Simpplr__Text_Intro__c": "This content is about human resources working to imporve people's happiness with the company.Our Financial services team is bringing in a lot of money and the CEO and executive teams are pushing things along. The IT team has also been doing a lot with new applications onboarded and training sessions every week."
            }
        ]
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['text_obj'])


if __name__ == '__main__':
    test_use_pipeline()
