import srt
import pandas as pd
import numpy as np
from eapl.df.eapl_kpi import eapl_df_agg_pct_kpis

try:
    from .nlp_glob import *
except ImportError:
    from nlp_glob import *


def eapl_srt_to_records(data, cfg):
    text_key = cfg['text_key']
    format = cfg.get('format', 'to_srt')
    ts_diff_thr = cfg.get('ts_diff_thr', 0.2)
    out_key = cfg['out_key']

    srt_text = data[text_key]
    subs = list(srt.parse(srt_text))

    # Extract ts and content
    td_1s = np.timedelta64(1, 's')
    subs_tup = [(s.index, s.start, s.end, s.content, s.proprietary) for s in subs]

    cols = ['s_index', 'start', 'end', 'content', 'proprietary']
    s_df = pd.DataFrame(subs_tup, columns=cols)

    s_df['prev_end'] = s_df.end.shift(1)
    s_df['diff'] = s_df['start'] - s_df['prev_end']
    s_df['diff'] = s_df['diff'] / td_1s
    s_df['diff'] = s_df['diff'].fillna(0)
    s_df['section_flag'] = np.where(s_df['diff'] > ts_diff_thr, 1, 0)
    s_df['new_index'] = s_df['section_flag'].cumsum()

    cfg = {
        'gpbycols': ['new_index'],
        'ops': [
            ['agg_feats', ('start', 'min', 'start')],
            ['agg_feats', ('end', 'max', 'end')],
            ['agg_feats', ('content', lambda sents: " ".join([s.strip() for s in sents]), 'content')],
        ]
    }
    c_srt = eapl_df_agg_pct_kpis(s_df, cfg)
    c_srt.rename(columns={'new_index': 'index'}, inplace=True)
    if format == 'records':
        c_srt = c_srt.to_dict(orient='records')

    if format == 'to_srt':
        c_srt['subtitles'] = c_srt.apply(lambda x: srt.Subtitle(x['index'], x['start'], x['end'], x['content']), axis=1)
        c_srt = list(c_srt['subtitles'])
        c_srt = srt.compose(c_srt)

    data[out_key] = c_srt
    return data


qg_gpt2_fmap = {
    'eapl_srt_to_records': eapl_srt_to_records,
}
nlp_func_map.update(qg_gpt2_fmap)


def test_srt_cfg():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_srt", required=True, help="Input srt file")
    ap.add_argument("-o", "--output_srt", required=False, help="Input srt file", default='output.srt')
    args = vars(ap.parse_args())

    with open(args['input_srt'], 'r') as fp:
        srt_text = fp.read()

    # list() is needed as srt.parse creates a generator

    data = {
        'srt_txt': srt_text
    }
    cfg = {
        'text_key': 'srt_txt',
        'format': 'to_srt',
        'out_key': 'srt_records'
    }
    data = eapl_srt_to_records(data, cfg)
    with open(args['output_srt'], 'w') as fp:
        fp.write(data['srt_records'])

    return None


if __name__ == '__main__':
    test_srt_cfg()
