import logging
from spacy.util import minibatch, compounding
from pathlib import Path
import copy

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs

nlp_p_logger = logging.getLogger(__name__)


def eapl_spacy_save_nlp(data, cfg):
    dir = Path(cfg.get('output_dir'))
    nlp_key = cfg.get('nlp_key', 'nlp')
    nlp = data[nlp_key]
    if not dir.exists():
        dir.mkdir()
        nlp.to_disk(dir)
        logging.debug(f"Saved model to {dir}")
    return data


def eapl_spacy_classifer_init(data, cfg):
    nlp_key = cfg.get('nlp_key', 'nlp')
    pipe_name = cfg.get('pipe_name', 'textcat')
    exclusive_classes = cfg.get('exclusive_classes', False)
    architecture = cfg.get('architecture', 'ensemble')
    nlp = data[nlp_key]
    if pipe_name not in nlp.pipe_names:
        textcat = nlp.create_pipe(pipe_name,
                                  config={"exclusive_classes": exclusive_classes, 'architecture': architecture})
        nlp.add_pipe(textcat, last=True)

    data[nlp_key] = nlp
    return data


def train_model_spacy(data, cfg):
    nlp_key = cfg.get('nlp_key', 'nlp')
    pipe_key = cfg.get('pipe_name', 'textcat')
    train_data_fmt = cfg.get('train_data_fmt', 'list')
    input_data_key = cfg.get('input_data_key', 'input_data_key')
    train_key = cfg.get('input_key', 'input_key')
    label_key = cfg.get('label_key', 'label_key')
    epoch = cfg.get('epoch', 30)
    drop = cfg.get('drop', 0.2)
    train_lst, label_list = [], []
    if train_data_fmt == 'records':
        for item in data[input_data_key]:
            tmp_txt = item[train_key]
            tmp_lable = item[label_key]
            label_list.extend(tmp_lable)
            train_lst.extend([tmp_txt] * len(tmp_lable))

    if train_data_fmt == 'list':
        train_lst = data[input_data_key][train_key]
        label_list = data[input_data_key][label_key]

    labels_default = dict((v, 0) for v in set(label_list))
    nlp = data[nlp_key]
    textcat = nlp.get_pipe(pipe_key)

    for i in list(set(label_list)):
        textcat.add_label(i)

    train_data = []

    for i in range(len(train_lst)):
        label_values = copy.deepcopy(labels_default)
        label_values[label_list[i]] = 1
        train_data.append((train_lst[i], {'cats': label_values}))

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != pipe_key]

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        logging.debug("Training model...")
        for i in range(epoch):
            losses = {}
            batches = minibatch(train_data, size=compounding(4, 32, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=drop, losses=losses)

            logging.debug(f"epoch  : {i} Loss : {losses['textcat']}")

    data[nlp_key] = nlp

    return data


def eapl_spacy_inference(data, text_dict, op_dict):
    doc_key = op_dict.get('doc_key', 'doc')
    out_key = op_dict.get('out_key')
    threshold = op_dict.get('threshold', 0.5)
    labels = text_dict[doc_key].cats
    labels_tmp = []
    for i in labels:
        if labels[i] > threshold:
            labels_tmp.append(i)
    text_dict[out_key] = labels_tmp
    return text_dict


def eapl_spacy_inference_batch(data, text_dict, op_dict):
    nlp_key = op_dict.get('nlp_key', 'nlp')
    input_lst_key = op_dict.get('input_lst', 'input_lst')
    out_key = op_dict.get('out_key')
    threshold = op_dict.get('threshold', 0.5)
    nlp = data[nlp_key]
    input_lst = text_dict[input_lst_key]
    output = []
    for each_txt in input_lst:
        doc = nlp(each_txt)
        lables = doc.cats
        lables_tmp = []
        for i in lables:
            if lables[i] > threshold:
                lables_tmp.append(i)

        output.append(lables_tmp)

    text_dict[out_key] = output
    return text_dict


eapl_spacy_classifier_func_map = {
    'eapl_spacy_save_nlp': eapl_spacy_save_nlp,
    'eapl_spacy_classifer_init': eapl_spacy_classifer_init,
    'train_model_spacy': train_model_spacy,
    'eapl_spacy_inference': eapl_spacy_inference,
    'eapl_spacy_inference_batch': eapl_spacy_inference_batch

}
nlp_func_map.update(eapl_spacy_classifier_func_map)


def train_classifier_spacy():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_pipe', 'init_pipe_clfr', 'train_model', 'save_model', 'load_model',
                       'records_label_scoring'],
        'init_pipe': {
            'func': 'eapl_nlp_pipeline_init',
        },

        'init_pipe_clfr': {
            'func': 'eapl_spacy_classifer_init',
            'exclusive_classes': False
        },

        'train_model': {
            'func': 'train_model_spacy',
            'train_data_fmt': 'records',
            'input_data_key': 'text_obj',
            'input_key': 'txt',
            'label_key': 'label',
            'epoch': 10,
            'pipe_name': 'textcat'
        },

        'save_model': {
            'func': 'eapl_spacy_save_nlp',
            'output_dir': 'spacy_tmp'

        },

        'load_model': {
            'func': 'eapl_nlp_pipeline_init',
            'model': 'spacy_tmp',
            'nlp_key': 'nlp_clf'
        },

        'records_label_scoring': {
            'func': 'eapl_nlp_record_process',
            'text_obj': 'test_obj',
            'ops': [
                {
                    "op": "create_spacy_doc",
                    'nlp_key': 'nlp_clf',
                    'txt_key': 'body',
                    'doc_key': 'test_doc_key'
                },
                {
                    "op": "eapl_spacy_inference",
                    'doc_key': 'test_doc_key',
                    'out_key': 'label',
                },
                {
                    "op": "manage_text_dict_keys",
                    "pop_keys": ['test_doc_key']
                },
            ]
        },
    }

    data_test = {
        'text_obj': [
            {
                'txt': 'what is my name?',
                'label': ["Question", "Personal"]
            },
            {
                'txt': 'what is your name?',
                'label': ["Question", "Personal"]
            },
            {
                'txt': 'My name is Vishwas.',
                'label': ["Statement"]
            }
        ],
        'test_obj': [
            {
                'body': 'Where is Bangalore?',
            },
            {
                'body': 'Weather in Bangalore is good.',
            }
        ],
        'input_lst_key': {
            'input_key': ["what is your name?", "what is my name?", "My name is vishwas", "I am from bangalore"],
            'label_key': ['Question', 'Question', 'Statement', 'Statement']
        }
    }
    func = nlp_func_map['eapl_data_process_fk_flow']
    data_test = func(data_test, nlp_cfg)
    pprint(data_test)
    pprint(data_test['nlp_clf']("what is your name?").cats)


if __name__ == '__main__':
    train_classifier_spacy()
