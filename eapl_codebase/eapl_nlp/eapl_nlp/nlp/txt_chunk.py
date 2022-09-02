from pprint import pprint
import logging

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs


def text_chunk_ntok_overlap(data, cfg):
    input_key = cfg.get('input_key', 'txt')
    output_key = cfg.get('output_key', input_key)
    format = cfg.get('format', 'list')
    num_tokens = cfg.get('num_tokens', 1000)
    overlap = cfg.get('overlap', )
    meta = cfg.get('meta', None)

    txt = data[input_key]
    txt_no_newl = txt.replace('\n', ' ')
    tokens = txt_no_newl.split()

    step_size = max(num_tokens - overlap, 1)
    chunks = [" ".join(tokens[i:i + num_tokens]) for i in range(0, len(tokens), step_size)]
    if format == 'list_dicts':
        chunk_ld = []
        for ch in chunks:
            dct = {'text': ch}
            if meta:
                dct.update({'meta': meta})
            chunk_ld.append(dct)
        chunks = chunk_ld

    data[output_key] = chunks
    return data


def text_chunk_ntok_overlap_ld(data, text_dict, op_dict):
    return text_chunk_ntok_overlap(text_dict, op_dict)


txt_chunk_fmap = {
    'text_chunk_ntok_overlap': text_chunk_ntok_overlap,
    'text_chunk_ntok_overlap_ld': text_chunk_ntok_overlap_ld,

}
nlp_func_map.update(txt_chunk_fmap)


def txt_overlap_chunking_folder(data_path, split_paragraphs=None):
    import os
    from glob import glob

    def read_file(fname):
        fdata = None
        if fname:
            with open(fname, 'r', encoding="utf8") as fp:
                fdata = fp.read()
        return fdata

    nlp_cfg = {
        'config_seq': ['txt_chunk_data'],

        'txt_chunk_data': {
            "func": "text_chunk_ntok_overlap",
            "input_key": "txt_data",
            "output_key": "txt_chunks_d",
            "num_tokens": 400,
            "overlap": 100,
            "format": "list_dicts"
        }
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    file_list = glob(f"{data_path}*.txt")[:]
    chunk_list = []
    for filepath in file_list:
        data = {
            "txt_data": read_file(filepath)
        }
        filename = os.path.basename(filepath)
        nlp_cfg['txt_chunk_data']['meta'] = {"name": filename}
        data = func(data, nlp_cfg)
        chunk_list = chunk_list + data["txt_chunks_d"]

    return chunk_list


def test_txt_chunking():
    txt_str = """
                    abcd xyz    def, 
                    cde fghhi
                    
                    
                    
                    
                    make this work,.
                    
                    """
    data = {
        'txt_ld': [
            {
                'txt': txt_str + " 1"
            },
            {
                'txt': txt_str + " 2"
            },

        ],

        'txt_data': txt_str
    }

    nlp_cfg = {
        'config_seq': ['txt_chunk_data', 'text_chunk_ld'],

        'txt_chunk_data': {
            "func": "text_chunk_ntok_overlap",
            "input_key": "txt_data",
            "output_key": "txt_chunks_d",
            "num_tokens": 10,
            "overlap": 9
        },

        'text_chunk_ld': {
            'func': 'eapl_nlp_record_process',
            'text_obj': 'txt_ld',
            'ops': [
                {
                    "op": "text_chunk_ntok_overlap_ld",
                    "meta": {"name": "file1.txt"},
                    "input_key": "txt",
                    "output_key": "txt_chunks",
                    "format": "list_dicts",
                    "num_tokens": 10,
                    "overlap": 9
                }
            ]

        }
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)


if __name__ == '__main__':
    test_txt_chunking()
    # overlap_chunk_list = txt_overlap_chunking_folder("D:/projects/uipath/pdfs/", split_paragraphs=None)
    # pprint(overlap_chunk_list[:5])
