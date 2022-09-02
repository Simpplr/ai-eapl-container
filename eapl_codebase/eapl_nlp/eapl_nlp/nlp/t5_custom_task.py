import pandas as pd
from simpletransformers.t5 import T5Model
import logging
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs

t5_hs_qg_p_logger = logging.getLogger(__name__)


def t5_hs_qg_create_training_data(data, cfg):
    train_data_ld_key = cfg.get("train_data_ld_key", "train_data_ld_key")
    input_text_key = cfg.get("input_text_key", "input_text")
    target_text_key = cfg.get("target_text_key", "target_text")
    train_pct = cfg.get("train_pct", 0.9)
    prefix = cfg.get("prefix", "hs_qg")
    io_pairs = data[train_data_ld_key]

    num_records = len(io_pairs)
    train_recs = int(num_records * train_pct)

    all_data = []
    for train_dct in io_pairs:
        input_text = train_dct[input_text_key]
        target_text = train_dct[target_text_key]
        tdata = [prefix, input_text, target_text]
        all_data.append(tdata)
    all_df = pd.DataFrame(all_data, columns=["prefix", "input_text", "target_text"])

    train_df = all_df.iloc[:train_recs, :]
    eval_df = all_df.iloc[train_recs:, :]
    return train_df, eval_df


def t5_train_custom_task_model(data, cfg):
    model_name = cfg.get('model_name', "t5-base")
    model_type = cfg.get('model_type', "t5")
    t5_model_key = cfg.get("t5_model_key", "t5_model_key")
    use_cuda = cfg.get("use_cuda", False)
    def_model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 128,
        "train_batch_size": 5,
        "num_train_epochs": 10,
        "save_model_every_epoch": False,
        "max_length": 20,
        "num_beams": 1,
    }
    model_args = cfg.get("model_args", def_model_args)

    t5_model_params = {
        "model_type": model_type,
        "model_name": model_name,
        "args": model_args,
        "use_cuda": use_cuda
    }
    train_df, eval_df = t5_hs_qg_create_training_data(data, cfg)

    # Create T5 Model
    model = T5Model(**t5_model_params)

    # Train T5 Model on new task
    model.train_model(train_df)

    # Evaluate T5 Model on new task
    model.eval_model(eval_df)
    data[t5_model_key] = model
    return data


def t5_load_custom_task_model(data, cfg):
    model_type = cfg.get("model_type", "t5")
    use_cuda = cfg.get("use_cuda", False)
    t5_model_key = cfg.get("t5_model_key", "t5_model_key")
    model_path = cfg.get('model_path', None)
    args = {
        "model_type": model_type,
        "args": {
            "multiprocessing_chunksize": 500
        },
        "use_cuda": use_cuda,
        "model_name": model_path
    }
    t5_model_param = cfg.get("t5_model_param", args)

    model = T5Model(**t5_model_param)

    data[t5_model_key] = model
    return data


def t5_predict_custom_task_td(data, text_dict, op_dict):
    t5_model_key = op_dict.get("t5_model_key", "t5_model_key")
    prefix = op_dict.get("prefix", "hs_qg")
    input_key = op_dict["input_key"]
    out_key = op_dict.get("out_key", "pred_txt")
    inp_str = text_dict[input_key]
    model = data[t5_model_key]

    text_dict[out_key] = model.predict([f"{prefix}: {inp_str}"])
    return text_dict


def t5_predict_custom_task(data, cfg):
    t5_model_key = cfg.get("t5_model_key", "t5_model_key")
    input_ld_key = cfg["input_ld_key"]
    prefix = cfg.get("prefix", "hs_qg")
    input_key = cfg["input_key"]
    out_key = cfg.get("out_key", "pred_txt")
    out_ld_key = cfg.get("out_ld_key", input_ld_key)
    model = data[t5_model_key]
    input_ld = data[input_ld_key]

    model_input = [f"{prefix}: {d[input_key]}" for d in input_ld]
    model_output = model.predict(model_input)
    for i, pred in enumerate(model_output):
        input_ld[i][out_key] = [pred]
    data[out_ld_key] = input_ld
    return data


t5_hs_qg_fmap = {
    't5_train_custom_task_model': t5_train_custom_task_model,
    't5_load_custom_task_model': t5_load_custom_task_model,
    't5_predict_custom_task_td': t5_predict_custom_task_td,
    't5_predict_custom_task': t5_predict_custom_task
}
nlp_func_map.update(t5_hs_qg_fmap)


def train_t5_new_task_model():
    from pprint import pprint

    data = {
        'train_ld': [
            {
                "input_text": "one",
                "target_text": "1"
            },
            {
                "input_text": "two",
                "target_text": "2"
            },
            {
                "input_text": "three",
                "target_text": "3"
            }, {
                "input_text": "four",
                "target_text": "4"
            },
        ],

        'test_ld': [
            {
                "input_text": "seven",
            },
            {
                "input_text": "eight",
            }
        ],

        'test2_ld': [
            {
                "input_text": "nine",
            },
            {
                "input_text": "ten",
            }
        ]
    }

    nlp_cfg = {
        'config_seq': ['t5_train_custom_task_model_cfg', 'record_nlp_ops', 'multiple_preds'],
        # 'config_seq': ['download_model', 'uncompress_file', 'load_custom_t5_model', 'record_nlp_ops_ptm'],

        't5_train_custom_task_model_cfg': {
            'func': 't5_train_custom_task_model',
            'train_data_ld_key': 'train_ld',
            "train_pct": 0.5,
            "t5_model_key": "t5_model_key",
            "model_args": {
                "reprocess_input_data": True,
                "overwrite_output_dir": True,
                "output_dir": '/tmp/sap_io_model',
                "max_seq_length": 128,
                "train_batch_size": 5,
                "num_train_epochs": 1,
                "save_model_every_epoch": False,
                "max_length": 20,
                "num_beams": 1,
            }
        },

        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'text_obj': 'test_ld',
            'ops': [
                {
                    "op": "t5_predict_custom_task_td",
                    "t5_model_key": "t5_model_key",
                    "input_key": "input_text"
                },
            ]
        },

        'multiple_preds': {
            'func': 't5_predict_custom_task',
            "t5_model_key": "t5_model_key",
            "input_ld_key": "test2_ld",
            "input_key": "input_text",
            "out_key": "target_text",
            "out_ld_key": "test2_out_ld"
        }
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)


def test_t5_hs_qg_model():
    from pprint import pprint
    try:
        from nlp_utils import eapl_config_import
    except:
        from .nlp_utils import eapl_config_import

    data = {
        'test_ld': [
            {
                "input_text": "SuccessFactors Learning Administration Help xyz Supported AICC Import and Export Files",
            }
        ]
    }

    nlp_cfg = {
        'config_seq': ["import_test", "uncompress_file", 'load_custom_t5_model',
                       'record_nlp_ops_ptm'],

        "import_test": {
            "func": "eapl_config_import",
            "imports": {
                "eapl_import": "from .eap_kpi_ref import eapl_kpi_non_df_ops"
            }
        },

        "download_model": {
            "func": "eapl_copy_file",
            "src_path": "https://s3.amazonaws.com/emplay.botresources/NLP_Models/sap_io_1.0.zip",
            "dest_path": "/tmp/"
        },

        "uncompress_file": {
            "func": "eapl_uc_file",
            "file_path": "/tmp/sap_io_1.0.zip",
            "out_path": "/tmp/"
        },

        'load_custom_t5_model': {
            'func': 't5_load_custom_task_model',
            't5_model_key': "t5_model_pretrained",
            'model_path': "/tmp/outputs",
        },

        'record_nlp_ops_ptm': {
            'func': 'eapl_nlp_record_process',
            'text_obj': 'test_ld',
            'ops': [
                {
                    "op": "t5_predict_custom_task_td",
                    "t5_model_key": "t5_model_pretrained",
                    "input_key": "input_text",
                    "prefix": "ques_gen"
                },
            ]
        },

    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['test_ld'])


if __name__ == '__main__':
    # train_t5_new_task_model()
    test_t5_hs_qg_model()
