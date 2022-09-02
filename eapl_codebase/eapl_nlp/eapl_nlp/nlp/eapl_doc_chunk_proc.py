import re
import json

try:
    from .nlp_glob import *
    from .nlp_utils import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_utils import *
    from nlp_ops import nlp_ops_funcs


def eapl_chunk_title_fix(data, cfg):
    input_key = cfg['input_key']
    input_format = cfg.get('input_format', 'list_of_records')
    output_format = cfg.get('output_format', input_format)
    output_key = cfg.get('output_key', input_key)
    title = cfg.get('title', 'Content Title')
    text = cfg.get('text', 'Content Text')
    th = cfg.get('title_hierarchy', 'Title Hierarchy')
    label = cfg.get('label', 'Label')
    heading_regex = cfg.get('heading_regex', r'^[\d]+\\s*-|^[\d.]+$')

    df = data[input_key]
    if input_format == 'list_of_records':
        df = pd.DataFrame.from_records(df)

    df = df.fillna('')
    df = df[df[text] != ''].reset_index(drop=True)  # Dropping the rows with empty content text
    df[title] = df.apply(lambda x: ' '.join(x[text].splitlines()[:2]) if (bool(re.search(heading_regex, x[title]))) else x[title],axis=1)
    df[title] = df.apply(lambda x: (x[title].split(' ', 1)[1]) if ((x[title].split(' ', 1)[0].isnumeric()) and (bool(re.search(heading_regex, x[title].split(' ')[1])))) else x[title], axis=1)
    df[title] = df.apply(lambda x: (x[text].splitlines()[2]) if ((len(x[text].splitlines()) > 2) and bool(re.search(r'\u2022', x[title]))) else x[title], axis=1)
    df[title] = df.apply(lambda x: (x[text].splitlines()[0]) if (x[title] == '') else x[title], axis=1)
    df[label] = df.apply(lambda x: x[title], axis=1)
    df[th] = df.apply(lambda x: ([x[title]]) if ((bool(re.search(heading_regex, x[title][0]))) or ((len(x[th])) == 0)) else x[th],axis=1)

    if output_format == 'list_of_records':
        df = df.to_dict(orient='records')

    data[output_key] = df
    return data


eapl_doc_chunk_proc_fmap = {
    "eapl_chunk_title_fix": eapl_chunk_title_fix
}
nlp_func_map.update(eapl_doc_chunk_proc_fmap)


def test_post_process_chunk():
    from pprint import pprint
    nlp_cfg = {
        "config_seq": ["import_funcs", "post_process_chunk"],
        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "eapl_doc_chunk_proc": "eapl_doc_chunk_proc"
            }
        },
        "post_process_chunk": {
            "func": "eapl_chunk_title_fix",
            "input_key": "input"
        }
    }

    # df = pd.read_csv("/home/amrutha/Downloads/Document_Chunking/doc_chunk.csv")
    f = open('/home/amrutha/Downloads/node_output.json')
    data_json = json.load(f)
    data = {
        'input': data_json
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data)

    return None


if __name__ == '__main__':
    test_post_process_chunk()
