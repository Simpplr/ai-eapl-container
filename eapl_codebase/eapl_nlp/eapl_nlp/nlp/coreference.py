from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import logging

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs

coref_res_logger = logging.getLogger(__name__)


def eapl_coref_allennlp_init(data, cfg):
    coref_key = cfg.get('coref_key', 'coref')
    path = cfg.get('coref_model_path',
                   "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
    predictor = nlp_glob.get(coref_key, None)
    if predictor is None:
        predictor = Predictor.from_path(path)
        nlp_glob[coref_key] = predictor

    data[coref_key] = predictor
    return data


def text_coref(data, text_dict, op_dict):
    doc_key = op_dict.get('doc_key', 'doc')
    coref_key = op_dict.get('out_key', 'text_coref_allennlp')
    predictor_key = op_dict.get('coref_key', 'coref')
    predictor = data[predictor_key]

    doc = text_dict[doc_key]
    coref_text = coref_res(doc, coref_key, predictor)
    text_dict[coref_key] = coref_text
    return text_dict


def coref_res(doc, coref_key, predictor):
    text = doc.text
    method = coref_key

    if method == "text_coref_allennlp":
        coref_text = predictor.coref_resolved(document=text)
    return coref_text


coref_res_fmap = {
    'eapl_coref_allennlp_init': eapl_coref_allennlp_init,
    'text_coreference': text_coref,
}
nlp_func_map.update(coref_res_fmap)


def test_coref_res():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_pipe', 'eapl_coref_allennlp_init', 'record_nlp_ops'],

        'init_pipe': {
            'func': 'eapl_nlp_pipeline_init',
            'coref_key': 'coref'
        },

        'eapl_coref_allennlp_init': {
            'func': 'eapl_coref_allennlp_init',
        },

        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    "op": "create_spacy_doc",
                },
                {
                    'op': 'text_coreference',
                    'doc_key': 'doc',
                    'out_key': 'text_coref_allennlp'
                },
                {
                    'op': 'manage_text_dict_keys',
                    'pop_keys': ['doc']
                }
            ]
        }
    }

    data = {
        'text_obj': [
            {
                "txt": """Sushant Singh Rajput passed away on June 14 at his Bandra residence. Since then, his fans and family members were demanding for a CBI inquiry. Recently, the Supreme Court announced its verdict and transferred the case to the CBI. The team of Central Bureau of Investigation is currently in Mumbai and they are actively investigating the actor’s death case. The CBI team recently questioned Sushant’s flatmate Siddharth Pithani and his cook Neeraj since they were present at the residence on June 14. Now according to the latest report by Times Now, Rhea Chakraborty’s brother Showik Chakraborty is currently being grilled by the CBI at the DRDO guesthouse."""
            },
            {
                "txt": """A single-day spike of 61,408 infections has taken India's coronavirus caseload to 31,06,348, data from the Union Health Ministry this morning showed. The number of deaths climbed to 57,542 with 836 people dying of the disease in a span of 24 hours. Recoveries among COVID-19 patients in the country surged to 23,38,035, pushing India's recovery rate to 75.27 per cent. Maharashtra accounts for the maximum of cases (6,82,383), followed by Tamil Nadu (3,79,385), Andhra Pradesh (3,53,111), Karnataka (2,77,814), Uttar Pradesh (1,87,781) and Delhi (1,61,466). India is the country with the third-highest coronavirus caseload in the world after the United States and Brazil."""
            },
        ]
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['text_obj'])

    return None


if __name__ == '__main__':
    test_coref_res()
