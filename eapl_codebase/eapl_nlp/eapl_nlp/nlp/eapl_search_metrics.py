import logging
from pytrec_eval import RelevanceEvaluator
import pandas as pd

try:
    from .nlp_glob import nlp_func_map, nlp_glob
    from .nlp_ops import nlp_ops_funcs
    from .nlp_utils import nlp_utils_fmap
except ImportError:
    from nlp_glob import nlp_func_map, nlp_glob
    from nlp_ops import nlp_ops_funcs
    from nlp_utils import nlp_utils_fmap

eapl_hs_logger = logging.getLogger(__name__)


def eapl_search_metrics(data, cfg):
    ref_data_key = cfg['ref_data_key']
    pred_data_key = cfg['pred_data_key']
    ref_id_key = cfg.get('ref_id_key', 'id')
    pred_id_key = cfg.get('pred_id_key', 'id')
    ref_query_key = cfg.get("ref_query_key", "query")
    pred_query_key = cfg.get("pred_query_key", "query")
    ref_score_key = cfg.get("ref_score_key", "score")
    pred_score_key = cfg.get("pred_score_key", "score")

    # Supported metrics : {'gm_map', 'relstring', 'num_rel', 'num_q', 'utility', 'map', 'iprec_at_recall',
    # 'G', 'gm_bpref', 'Rprec_mult', 'num_nonrel_judged_ret', 'num_rel_ret', 'recall', 'set_recall', 'runid',
    # 'ndcg_rel', 'num_ret', 'map_cut', 'success', 'ndcg_cut', 'set_P', '11pt_avg', 'Rprec', 'recip_rank', 'set_map',
    # 'set_relative_P', 'relative_P', 'Rndcg', 'P', 'infAP', 'bpref', 'binG', 'ndcg', 'set_F'}
    eval_type_list = set(cfg['eval_type_list'])
    out_key = cfg['out_key']

    ref_data = data[ref_data_key]
    pred_data = data[pred_data_key]

    ref_data = pd.DataFrame(ref_data).explode(ref_id_key).to_dict('records')

    qrel = {}
    run_dict = {}

    for each_rec in ref_data:
        query = each_rec[ref_query_key]
        qrel[query] = qrel.get(query, {})
        id_score_dct = {each_rec[ref_id_key]: each_rec[ref_score_key]}
        qrel[query].update(id_score_dct)

    for each_rec in pred_data:
        query = each_rec[pred_query_key]
        run_dict[query] = run_dict.get(query, {})
        id_score_dct = {each_rec[pred_id_key]: each_rec[pred_score_key]}
        run_dict[query].update(id_score_dct)

    evaluator = RelevanceEvaluator(qrel, eval_type_list)
    metrics = evaluator.evaluate(run_dict)
    metrics['avrg_total'] = pd.DataFrame(metrics).mean(axis=1).to_dict()

    data[out_key] = metrics
    return data


search_eval_func = {
    "eapl_search_metrics": eapl_search_metrics

}
nlp_func_map.update(search_eval_func)


def test_search_eval():
    from pprint import pprint
    data = {
        "eval_dict": [
            {
                "id": ["d1"],
                "score": 5,
                "query": "check tool"

            },
            {
                "id": ["d2"],
                "score": 1,
                "query": "api not working"
            }
        ],
        "searchResult": [
            {
                "id": "d2",
                "score": 2.0,
                "query": "check tool"

            },
            {
                "id": "d1",
                "score": 1.0,
                "query": "check tool"

            },
            {
                "id": "d2",
                "score": 2.0,
                "query": "api not working"
            }
        ],
        "idp_ref": [
            {
                "query": "How to access pending data in EC Workflows with API's?",
                "id": ["sapio_idp_133", "sapio_idp_137", "sapio_idp_143"]
            }
        ]
    }

    nlp_cfg = {
        'config_seq': ['eval_search'],
        'eval_search': {
            'func': 'eapl_search_metrics',
            'ref_data_key': 'eval_dict',
            'pred_data_key': 'searchResult',
            'ref_id_key': 'id',
            'pred_id_key': 'id',
            'eval_type_list': ['map', 'ndcg', 'ndcg_rel'],
            'out_key': 'metrics'
        }
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)


if __name__ == '__main__':
    test_search_eval()
