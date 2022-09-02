import pandas as pd
import numpy as np
import random
import pickle
import string
import time
import logging
import urllib.parse
import re
import datetime
from datetime import timezone
from pandas.api.types import is_string_dtype, is_numeric_dtype
from collections import Counter

eapl_kpi_logger = logging.getLogger(__name__)


def eapl_convert_to_list(x):
    return x if isinstance(x, list) else [x]


def eapl_perf_logger(pdict, func, time_event):
    if time_event == 'start':
        pdict.update({func: {}})
        pdict[func].setdefault('st', time.time())

    elif time_event == 'end':
        pdict[func].setdefault('et', time.time())
        pdict[func].setdefault('tt', int((pdict[func]['et'] - pdict[func]['st']) * 1000))

    return pdict


def eapl_remove_elems(a, b, n=3):
    out = np.NaN
    if isinstance(b, list) and isinstance(a, list):
        out = [bi for bi in b if bi not in a]
        out = out[:n]
    return out


def eapl_concat_cols(df, group_cols, mapped_col, sep=',', drop=False):
    gc, mc = group_cols, mapped_col
    gc = eapl_convert_to_list(gc)

    if len(gc) > 1:
        df[mc] = df[gc[0]].str.cat(df[gc[1:]], sep=sep)
    else:
        df[mc] = df[gc[0]]

    if drop:
        df.drop(columns=gc, inplace=True)
    return df


def eapl_map_typecast(df, col_map=None, num_cols=None, dt_cols=None, dt_fmt=None):
    if col_map:
        df.rename(columns=col_map, inplace=True)

    if num_cols:
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if dt_cols:
        for col in dt_cols:
            df[col] = pd.to_datetime(df[col], format=dt_fmt, errors='coerce')

    return df


def eapl_clean_string(txt_col):
    punct = string.punctuation + ' '
    punct = punct.replace('|', '')
    transtab = str.maketrans(dict.fromkeys(punct, ''))
    return txt_col.astype(str).str.translate(transtab)


# Filter data
def eapl_filter_data_wrapper(data, config):
    df = data[config['input_df']]

    if 'select_cols' in config.keys():
        df = df[config['select_cols']]

    if 'drop_cols' in config.keys():
        df.drop(labels=config['drop_cols'], axis=1, inplace=True)

    if 'sel_rows_query' in config.keys():
        for query in config['sel_rows_query']:
            df.query(query, inplace=True)

    data[config['output_df']] = df
    return data


def eapl_dedup(df, filt, pc, sc, rc, agg_type):
    return eapl_df_agg_rank_kpis(df, None, [pc], filt, sc, rc, agg_type, sc, 1)


# Creation of dedup module
def eapl_dedup_wrapper(data, config):
    df = data[config['input_df']]

    for dedup_type in ['dedup_cols_by_filt_val_df', 'dedup_cols_by_filt_val_dict']:
        if dedup_type in config.keys():
            for filt, pc, sc, rc, agg_type, out in config[dedup_type]:
                map_df = eapl_dedup(df, filt, pc, sc, rc, agg_type)
                data[out] = dict(zip(map_df[pc], map_df[sc])) if dedup_type == 'dedup_cols_by_filt_val_dict' else map_df

    return data


# DataFrame summary
def eapl_col_value_counts(data, cfg):
    df_list = cfg.get('tbl_list', [])
    unq_count_thr = cfg.get('unq_count_thr', 100)

    val_count_all_df = pd.DataFrame()
    for dfname in df_list:
        df = data[dfname]
        for col in df.columns:
            unq_count = df[col].nunique()
            if unq_count <= unq_count_thr:
                val_count_df = df[col].value_counts(dropna=False).reset_index()
                val_count_df.columns = ['colval', 'count']
                val_count_df['count_pct'] = val_count_df['count'] / val_count_df['count'].sum()
                val_count_df['tbl'] = dfname
                val_count_df['col'] = col
                val_count_df = val_count_df[['tbl', 'col', 'colval', 'count', 'count_pct']]
                val_count_all_df = pd.concat([val_count_all_df, val_count_df], axis=0)

    data[cfg['output_df']] = val_count_all_df
    return data


# DataFrame summary
def eapl_attribute_kpi_summary(data, cfg):
    df_list = cfg.get('tbl_list', [])
    unq_count_thr = cfg.get('unq_count_thr', 100)

    kpi_cfg = cfg.copy()
    kpi_all_df = pd.DataFrame()
    for dfname in df_list:
        df = data[dfname]
        for col in df.columns:
            unq_count = df[col].nunique()
            if unq_count <= unq_count_thr:
                kpi_cfg['gpbycols'] = [col]
                kpi_df = eapl_df_agg_pct_kpis(df, kpi_cfg)
                kpi_df.rename(columns={col: 'colval'}, inplace=True)
                kpi_df.insert(0, 'col', col)
                kpi_df.insert(0, 'tbl', dfname)
                kpi_all_df = pd.concat([kpi_all_df, kpi_df], axis=0)

    kpi_all_df = kpi_all_df.query("colval == colval")
    data[cfg['output_df']] = kpi_all_df
    return data


def eapl_datasize_summary_wrapper(data, cfg):
    df_shape = data[cfg['input_df']].shape
    shape_dict = {'Rows_Count': [df_shape[0]], 'Columns_Count': [df_shape[1]]}
    data[cfg['output_df']] = pd.DataFrame.from_dict(shape_dict)
    return data


def eapl_compute_outlier_pct(df_col, iqr_mul=3):
    quartile_1, quartile_3 = np.nanpercentile(df_col, q=[25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * iqr_mul)
    upper_bound = quartile_3 + (iqr * iqr_mul)
    return np.round(len(df_col[(df_col > upper_bound) | (df_col < lower_bound)]) * 100 / len(df_col), 4)


def eapl_top_bot_n_categories(df_col, n=5, top=True):
    try:
        df_col_clean = df_col.dropna()
        c = Counter(df_col_clean)

        if top:
            val_freq = [(i, str(c[i]), str(np.round(c[i] / len(df_col) * 100.0, 4))) for i, count in c.most_common(n)]
        else:
            val_freq = [(i, str(c[i]), str(np.round(c[i] / len(df_col) * 100.0, 4))) for i, count in
                        c.most_common()[-1:-n - 1:-1]]

        str_n_cats = [':'.join(tup) for tup in val_freq]
        str_cats = ', '.join(str_n_cats)
    except Exception as e:
        eapl_kpi_logger.error(repr(e))
        str_cats = 'Processing Error'

    return str_cats


def eapl_data_summary(df):
    nrows, ncols = df.shape
    df_summary = df.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99], include='all').T
    df_summary['type'] = df.dtypes
    df_summary['NA_pct'] = ((nrows - df_summary['count']) * 100) / nrows

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_nanall_cols = set(df_summary.query("count >0").index)
    num_cols = set(num_cols).intersection(non_nanall_cols)
    for col in num_cols:
        df_summary.loc[col, "box_3iqr_outlier_pct"] = eapl_compute_outlier_pct(df[col])
        df_summary.loc[col, "zero_neg_pct"] = (np.sum((df[col] <= 0).values.ravel()) * 100) / nrows

    obj_cols = df.select_dtypes(include='object').columns.tolist()
    for col in obj_cols:
        df_summary.loc[col, "top5_category"] = eapl_top_bot_n_categories(df[col])
        df_summary.loc[col, "bot5_category"] = eapl_top_bot_n_categories(df[col], top=False)

    return df_summary


def eapl_data_summary_wrapper(data, cfg):
    df = data[cfg['input_df']]
    if cfg.get('input_df_filt', False):
        df = df.query(cfg['input_df_filt'])
    data[cfg['output_df']] = eapl_data_summary(df)
    return data


def eapl_groupby_data_summary(data, cfg):
    df = data[cfg['input_df']]
    if cfg.get('input_df_filt', False):
        df = df.query(cfg['input_df_filt'])
    groupby_cols = cfg['gpbycols']
    sel_cols = df.columns
    if cfg.get('sel_cols', False):
        sel_cols = list(set(cfg['sel_cols']).union(set(groupby_cols)))

    data[cfg['output_df']] = df[sel_cols].groupby(groupby_cols).describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99], include='all')
    return data


# Derived columns
def eapl_get_attribute(col, attr, attr_args=None, attr_kwargs=None):
    attr_split = attr.split('.')
    out = col
    for x in attr_split:
        out = getattr(out, x)
    return out


def eapl_convert_to_set(val):
    if isinstance(val, (list, set)):
        out = set(val)
    elif isinstance(val, str):
        out = set([val])
    else:
        out = set()
    return out


def eapl_diff_set_n(l_list, r_list, n):
    out = list(eapl_convert_to_set(l_list) - eapl_convert_to_set(r_list))[:n]
    out = out if len(out) else np.NaN
    return out[0] if (isinstance(out, list) and n == 1) else out


def eapl_create_map(df_col, map_dict):
    unq_vals = list(df_col.unique())
    unq_vals_dict = dict(zip(unq_vals, unq_vals))
    for k, v in map_dict.items():
        unq_vals_dict[k] = v
    return unq_vals_dict


def eapl_derived_col_op_args(df, op, args):
    eapl_kpi_logger.debug('DERIVED COL OP : %s ARGS: %s', op, args)

    if 'df_getattr' == op:
        name, params = args
        df = getattr(df, name)(**params)

    elif 'df_col_getattr' == op:
        src_col, dst_col, name, params = args
        df[dst_col] = getattr(df[src_col], name)(**params)

    elif 'fix_col_names' == op:
        df.rename(columns=dict(zip(df.columns, df.columns.str.replace(" ", "_"))), inplace=True)

    elif 'rename_cols' == op:
        col_map = args
        df.rename(columns=col_map, inplace=True)

    elif 'fix_col_case' == op:
        case_func = args
        df.columns = map(case_func, df.columns)

    elif 'drop_cols' == op:
        df.drop(columns=args, inplace=True)

    elif 'tc_dt_col' == op:
        col = args
        df[col] = pd.to_datetime(df[col], errors='coerce')

    elif 'tc_dt_fmt_col' == op:
        col, fmt = args
        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')

    elif 'tc_num_col' == op:
        col = args
        df[col] = pd.to_numeric(df[col], errors='coerce')

    elif 'tc_num_nonstd_col' == op:
        col = args
        df[col] = pd.to_numeric(df[col].str.replace(r"[$,'%]+", ""), errors='coerce')

    elif 'dict_bad_vals_to_na' == op:
        df.replace(args, inplace=True)

    elif 'list_bad_vals_to_na' == op:
        badval_dict = {bd: np.NaN for bd in args}
        df.replace(badval_dict, inplace=True)

    elif 'assign_cols' == op:
        dst_col, value = args
        df[dst_col] = value

    elif 'assign_list_val' == op:
        dst_col, value = args
        df[dst_col] = df.apply(lambda x: value, axis=1)

    elif op in ['map_values', 'sel_map_values']:
        src_col, dst_col, mapping, null_value = args
        if 'sel_map_values' == op:
            mapping = eapl_create_map(df[src_col], mapping)
        df[dst_col] = df[src_col].map(mapping, na_action='ignore').fillna(null_value)

    elif 'join_cols' == op:
        src_cols, dst_col = args
        df[dst_col] = df[src_cols].apply(lambda x: '_'.join(list(map(str, x))), axis=1)

    elif 'cols_to_list' == op:
        src_cols, dst_col = args
        df[dst_col] = df[src_cols].apply(lambda x: [x[col] for col in x.columns if x[col] is not None], axis=1)

    elif 'list_to_str' == op:
        src_col, dst_col, join_str = args
        df[dst_col] = df[src_col].apply(lambda x: join_str.join(x) if isinstance(x, list) else None)

    elif 'attr_cols' == op:
        src_col, dst_col, attr_str = args
        df[dst_col] = eapl_get_attribute(df[src_col], attr_str)

    elif 'attr_arg_cols' == op:
        src_col, dst_col, attr_str, attr_args, attr_kwargs = args
        df[dst_col] = getattr(df[src_col], attr_str)(*attr_args, **attr_kwargs)

    elif 'bin_labels' == op:
        src_col, dst_col, bin_list, label_list = args
        df[dst_col] = pd.cut(df[src_col], bins=bin_list, labels=label_list, include_lowest=True).astype('object')

    elif 'rank' == op:
        src_col, dst_col, method, asc, pct = args
        df[dst_col] = df[src_col].rank(pct=pct, method=method, na_option='bottom', ascending=asc)

    elif 'pct_bin_labels' == op:
        gpbycols, src_col, dst_col, bin_list, label_list = args
        if isinstance(gpbycols, (list, str)):
            pct_col = df.groupby(gpbycols)[src_col].rank(pct=True)
        else:
            pct_col = df[src_col].rank(pct=True)
        df[dst_col] = pd.cut(pct_col, bins=bin_list, labels=label_list, include_lowest=True).astype('object')

    elif 'round_cols' == op:
        src_col, dst_col, dp = args
        df[dst_col] = df[src_col].round(dp)

    elif op in ['np_where_cols', 'np_where_cols_cc', 'np_where_cols_cv', 'np_where_cols_vc', 'np_where_cols_vv',
                'np_where_eval_cc', 'np_where_eval_cv', 'np_where_eval_vc', 'np_where_eval_vv']:
        cond_col_expr, true_col, false_col, dst_col = args
        if op in ['np_where_cols', 'np_where_cols_cc', 'np_where_cols_cv', 'np_where_cols_vc', 'np_where_cols_vv']:
            cond_col = df[cond_col_expr]
        else:
            cond_col = df.eval(cond_col_expr)

        if op in ['np_where_cols', 'np_where_cols_cc', 'np_where_eval_cc']:
            df[dst_col] = np.where(cond_col, df[true_col], df[false_col])
        elif op in ['np_where_cols_cv', 'np_where_eval_cv']:
            df[dst_col] = np.where(cond_col, df[true_col], false_col)
        elif op in ['np_where_cols_vc', 'np_where_eval_vc']:
            df[dst_col] = np.where(cond_col, true_col, df[false_col])
        elif op in ['np_where_cols_vv', 'np_where_eval_vv']:
            df[dst_col] = np.where(cond_col, true_col, false_col)

    elif 'append_msgs' == op:
        src_cols, msg, dst_col = args
        df[dst_col] = df[src_cols].apply(lambda x: msg % tuple(x) if x.notnull().all() else np.NaN, axis=1)

    elif 'cond_append_msgs' == op:
        src_cols, msg, dst_col, cond_col, cond_val = args
        df[dst_col] = df[src_cols].apply(lambda x: msg % tuple(x) if x.notnull().all() else np.NaN, axis=1)
        df[dst_col] = np.where(df[cond_col] == cond_val, df[dst_col], None)

    elif 'concat_cols_str' == op:
        # Need enhancements to support non string columns and np.NaN values
        src_cols, join_str, dst_col = args
        src_cols = [col for col in src_cols if col in df.columns]
        df[dst_col] = df[src_cols].stack().groupby(level=0).apply(join_str.join)

    elif 'random_choice' == op:
        src_col, dst_col = args
        df[dst_col] = df[src_col].apply(lambda x: random.choice(x) if (isinstance(x, list) and len(x) > 0) else np.NaN)

    elif 'diff_dt_unit' == op:
        ldt_col, rdt_col, unit, dst_col = args
        df[dst_col] = eapl_get_attribute(df[ldt_col] - df[rdt_col], unit)

    elif 'diff_sets' == op:
        lcol, rcol, dst_col = args
        df[dst_col] = df.apply(lambda x: list(eapl_convert_to_set(x[lcol]) - eapl_convert_to_set(x[rcol])),
                               axis=1)

    elif 'intersect_sets' == op:
        lcol, rcol, dst_col = args
        df[dst_col] = df.apply(lambda x: list(eapl_convert_to_set(x[lcol]).intersection(eapl_convert_to_set(x[rcol]))),
                               axis=1)

    elif 'diff_sets_n' == op:
        lcol, rcol, dst_col, n = args
        df[dst_col] = df.apply(lambda x: eapl_diff_set_n(x[lcol], x[rcol], n), axis=1)

    elif 'val_in_list' == op:
        lcol, rcol, dst_col = args
        df[dst_col] = df.apply(lambda x: x[lcol] in (x[rcol] if isinstance(x[rcol], list) else []), axis=1)

    elif 'str_in_list' == op:
        val_str, rcol, dst_col = args
        df[dst_col] = df.apply(lambda x: val_str in (x[rcol] if isinstance(x[rcol], list) else []), axis=1)

    elif 'str_contains' == op:
        src_col, ch_str, dst_col = args
        df[dst_col] = df[src_col].str.contains(ch_str, regex=False, na=False, case=False)

    elif 'date_offset' == op:
        src_col, offset, unit, dst_col = args
        df[dst_col] = df[src_col] + pd.to_timedelta(offset, unit=unit)

    elif 'eval_cols' == op:
        eval_str = args
        df.eval(eval_str, inplace=True)

    elif 'nan_fill_val' == op:
        col, nan_fill_val = args
        df[col].fillna(value=nan_fill_val, inplace=True)

    elif op in ['map_typecast', 'map_typecast_sc']:
        map_col, num_cols, dt_cols, dt_fmt = args
        if op == 'map_typecast_sc':
            cols = list(map_col.keys())
            df = df[cols]
        df = eapl_map_typecast(df, map_col, num_cols=num_cols, dt_cols=dt_cols, dt_fmt=dt_fmt)

    elif 'concat_columns' == op:
        gc, mc, sep, drop = args
        df = eapl_concat_cols(df, gc, mc, sep=sep, drop=drop)

    elif 'encode_decode' == op:
        src_col, encode, decode, dst_col = args
        df[dst_col] = df[src_col].str.encode(encode).str.decode(decode)

    elif 'decode_encode' == op:
        src_col, decode, encode, dst_col = args
        df[dst_col] = df[src_col].str.decode(decode).str.encode(encode)

    elif 'encode' == op:
        src_col, encode, dst_col = args
        df[dst_col] = df[src_col].str.encode(encode)

    elif 'decode' == op:
        src_col, decode, dst_col = args
        df[dst_col] = df[src_col].str.decode(decode)

    elif 'replace_strip' == op:
        src_col, rep_from, rep_to, dst_col = args
        df[dst_col] = df[src_col].str.replace(rep_from, rep_to).str.strip()

    elif 'strip' == op:
        src_col, strip_type, strip_str, dst_col = args
        if strip_type == 'lstrip':
            df[dst_col] = df[src_col].str.lstrip(strip_str).str.strip()
        elif strip_type == 'rstrip':
            df[dst_col] = df[src_col].str.rstrip(strip_str).str.strip()
        elif strip_type == 'strip':
            df[dst_col] = df[src_col].str.strip(strip_str).str.strip()

    elif 'filter' == op:
        filter_str, reset_index = args
        df = df.query(filter_str).reset_index(drop=reset_index)

    elif 'explode' == op:
        exp_col, reset_index = args
        df = df.explode(exp_col).reset_index(drop=reset_index)

    elif 'drop_duplicates' == op:
        subset = args
        if subset:
            df = df.drop_duplicates(subset=subset)
        else:
            df = df.drop_duplicates()

    elif 'custom_funcs' == op:
        func = args
        df = func(df)

    elif 're_ext_pat' == op:
        org_col_name, new_col_name, pattern, group_num = args
        df[new_col_name] = df[org_col_name].apply(lambda item: re.search(pattern, item).group(group_num))

    elif 'split_col' == op:
        org_col_name, new_col_name, delimeter, pos = args
        df[new_col_name] = df[org_col_name].apply(lambda item: item.split(delimeter)[pos])

    elif 'epoch_datetime' == op:
        org_col_name, new_col_name, format, timezone_name = args
        if timezone_name == "UTC":
            df[new_col_name] = df[org_col_name].apply(lambda item: datetime.datetime.strptime(item, format))
            df[new_col_name] = df[new_col_name].apply(lambda item: item.replace(tzinfo=timezone.utc).timestamp())

    elif 'url_encode' == op:
        src_col, dst_col, encoding, safe = args
        if encoding in ["UTF-8", None]:
            df[dst_col] = df[src_col].apply(lambda item: urllib.parse.quote_plus(item, safe=safe))

    elif 'sort_values' == op:
        by, ascending, key = args
        df = df.sort_values(by=by, ascending=ascending, key=key)

    else:
        eapl_kpi_logger.warning('No such column level operation')

    return df


def eapl_derived_cols(df, config):
    for op, args in config:
        try:
            eapl_kpi_logger.debug('Processing OP : %s ARGS: %s', op, args)
            df = eapl_derived_col_op_args(df, op, args)
        except Exception as e:
            eapl_kpi_logger.error('**FAILED** OP : %s ARGS: %s', op, args)
            eapl_kpi_logger.error(f"Exception: {e}")

    return df


def eapl_derived_cols_wrapper(data, cfg):
    data[cfg['output_df']] = eapl_derived_cols(data[cfg['input_df']], cfg['ops'])
    return data


# Data Join
def eapl_df_join_wrapper(data, config):
    left_df = data[config['left_df']]
    if config['left_df_cols'] is None:
        left_df_cols = list(data[config['left_df']].columns)
    else:
        left_df_cols = config['left_df_cols']
    right_df = data[config['right_df']]
    if config['right_df_cols'] is None:
        right_df_cols = list(data[config['right_df']].columns)
    else:
        right_df_cols = config['right_df_cols']
    left_on = config['left_on']
    right_on = config['right_on']
    how = config['how']
    data[config['output_df']] = left_df[left_df_cols].merge(right_df[right_df_cols], left_on=left_on, right_on=right_on,
                                                            how=how)
    return data


# KPI creation based on Aggregration
def eapl_kpi_merge(feat_df, aggc_df, merge_type='outer', na_val=None):
    if feat_df is None:
        feat_df = aggc_df
    else:
        aggc_cols = list(set(aggc_df.select_dtypes(include=[np.number]).columns) - set(feat_df.columns))
        feat_df = feat_df.merge(aggc_df, how=merge_type)
        if na_val is not None:
            for col in aggc_cols:
                nv = pd.Timedelta(seconds=na_val) if feat_df[col].dtype == "timedelta64[ns]" else na_val
                feat_df[col] = feat_df[col].fillna(value=nv)
    return feat_df


def eapl_df_filt_agg_pct_kpis(df, feat_df, gpbycols, agg_type, filt, aggcol, aggop, mc):
    if filt is None:
        aggc_df = df.groupby(gpbycols)[aggcol].agg(aggop).reset_index()
    else:
        aggc_df = df.query(filt).groupby(gpbycols)[aggcol].agg(aggop).reset_index()
    aggc_df.rename(columns={aggcol: mc}, inplace=True)

    if agg_type in ['agg_feats_pct', 'agg_feats_rpct', 'filt_agg_feats_pct', 'filt_agg_feats_rpct']:
        aggc_df[mc + '_pct'] = aggc_df[mc] / aggc_df[mc].sum()
        if agg_type in ['agg_feats_rpct', 'filt_agg_feats_rpct']:
            f, u = pd.factorize(aggc_df[gpbycols[0]].values)
            w = aggc_df[mc].values
            aggc_df = aggc_df.assign(rpct=w / np.bincount(f, w)[f])
            aggc_df.rename(columns={'rpct': mc + '_rpct'}, inplace=True)
    feat_df = eapl_kpi_merge(feat_df, aggc_df, na_val=0)
    return feat_df


def eapl_df_agg_rank_kpis(df, feat_df, gpbycols, filt, oc, rc, agg_type, mc, topn):
    if filt is None:
        aggc_df = df.groupby(gpbycols + [oc]).agg({rc: agg_type}).reset_index()
    else:
        aggc_df = df.query(filt).groupby(gpbycols + [oc]).agg({rc: agg_type}).reset_index()

    aggc_df['rank'] = aggc_df.groupby(gpbycols)[rc].rank(method='first', ascending=False, na_option='bottom')
    aggc_df = aggc_df.query("rank <= @topn", local_dict={'topn': topn})
    if topn > 1:
        aggc_df = aggc_df.groupby(gpbycols, as_index=False).apply(lambda x: x.sort_values('rank'))
        aggc_df = aggc_df.groupby(gpbycols)[oc].apply(list).reset_index()
    aggc_df = aggc_df.rename(columns={oc: mc})[gpbycols + [mc]]
    feat_df = eapl_kpi_merge(feat_df, aggc_df)
    return feat_df


def eapl_df_agg_colval_kpis(df, feat_df, gpbycols, filt, oc, rc, agg_type, mc, kpitype='both', max_flag=False):
    if filt is None:
        aggc_df = df.groupby(gpbycols + [oc]).agg({rc: agg_type}).reset_index()
    else:
        aggc_df = df.query(filt).groupby(gpbycols + [oc]).agg({rc: agg_type}).reset_index()

    aggc_df[oc] = mc + '_' + eapl_clean_string(aggc_df[oc])

    if kpitype in ['val', 'both']:
        agg_df_pvt = pd.pivot_table(aggc_df, values=rc, index=gpbycols, columns=oc, aggfunc=np.sum)
        if max_flag:
            agg_df_pvt['max_' + mc + '_lvl'] = agg_df_pvt.max(axis=1, numeric_only=True)
        feat_df = eapl_kpi_merge(feat_df, agg_df_pvt.reset_index(), na_val=0)

    if kpitype in ['pct', 'both']:
        level = list(range(len(gpbycols)))
        aggc_gp_df = aggc_df.groupby(gpbycols + [oc])[rc].sum()
        aggc_pct_df = aggc_gp_df.groupby(level=level).apply(lambda x: x / x.sum()).reset_index()
        aggc_pct_df[oc] = 'pct_' + aggc_pct_df[oc]
        aggc_pct_df_pvt = pd.pivot_table(aggc_pct_df, values=rc, index=gpbycols, columns=oc, aggfunc=np.sum)
        if max_flag:
            aggc_pct_df_pvt['max_pct_' + mc + '_lvl'] = aggc_pct_df_pvt.max(axis=1, numeric_only=True)
        feat_df = eapl_kpi_merge(feat_df, aggc_pct_df_pvt.reset_index(), na_val=0)

    return feat_df


def eapl_filt_summary_feats(df, feat_df, gpbycols, args):
    filt, aggcol, oplist, mc = args
    if oplist is None:
        if is_numeric_dtype(df[aggcol]):
            oplist = ['count', 'mean', 'min', 25, 'median', 75, 'max']
        elif is_string_dtype(df[aggcol]):
            oplist = ['count', 'nunique']

    for op in oplist:
        aggop, op_mc = op, '%s_%s' % (op, mc)
        if isinstance(op, (int, float)):
            aggop, op_mc = lambda x: np.nanpercentile(x, q=op), 'p%d_%s' % (op, mc)

        feat_df = eapl_df_filt_agg_pct_kpis(df, feat_df, gpbycols, 'filt_agg_feats', filt, aggcol, aggop, op_mc)

    return feat_df


def eapl_df_agg_pct_kpis(df, config):
    if config.get('input_df_filt', False):
        df = df.query(config['input_df_filt'])

    gpbycols = config['gpbycols']
    feat_df = df[gpbycols].drop_duplicates()

    for agg_type, args in config['ops']:
        try:
            eapl_kpi_logger.debug('AGG TYPE : %s ARGS: %s', agg_type, args)
            if agg_type in ['agg_feats', 'agg_feats_pct', 'agg_feats_rpct']:
                aggcol, aggop, mc = args
                feat_df = eapl_df_filt_agg_pct_kpis(df, feat_df, gpbycols, agg_type, None, aggcol, aggop, mc)

            elif agg_type in ['filt_agg_feats', 'filt_agg_feats_pct', 'filt_agg_feats_rpct']:
                filt, aggcol, aggop, mc = args
                feat_df = eapl_df_filt_agg_pct_kpis(df, feat_df, gpbycols, agg_type, filt, aggcol, aggop, mc)

            elif agg_type == 'prim_feats':
                oc, rc, agg_type, mc = args
                feat_df = eapl_df_agg_rank_kpis(df, feat_df, gpbycols, None, oc, rc, agg_type, mc, 1)

            elif agg_type == 'filt_prim_feats':
                filt, oc, rc, agg_type, mc = args
                feat_df = eapl_df_agg_rank_kpis(df, feat_df, gpbycols, filt, oc, rc, agg_type, mc, 1)

            elif agg_type == 'filt_topn_feats':
                filt, oc, rc, agg_type, mc, topn = args
                feat_df = eapl_df_agg_rank_kpis(df, feat_df, gpbycols, filt, oc, rc, agg_type, mc, topn)

            elif agg_type == 'filt_colval_feats':
                filt, oc, rc, agg_type, mc, kpitype = args
                feat_df = eapl_df_agg_colval_kpis(df, feat_df, gpbycols, filt, oc, rc, agg_type, mc, kpitype)

            elif agg_type == 'filt_summary_feats':
                feat_df = eapl_filt_summary_feats(df, feat_df, gpbycols, args)

            else:
                feat_df = eapl_derived_col_op_args(feat_df, agg_type, args)

        except Exception as e:
            eapl_kpi_logger.debug('**FAILED** AGG TYPE : %s ARGS: %s', agg_type, args)
            eapl_kpi_logger.debug(f"Exception: {e}")

    return feat_df


def eapl_df_agg_pct_kpis_wrapper(data, cfg):
    data[cfg['output_df']] = eapl_df_agg_pct_kpis(data[cfg['input_df']], cfg)
    return data


def eapl_modify_config(inp_config, dict_mods):
    out_config = inp_config.copy()
    for key in dict_mods.keys():
        out_config[key] = dict_mods[key]

    return out_config


def eapl_data_process_flow(data, config):
    for cfg_name in config['config_seq']:
        eapl_kpi_logger.info("Processing Config - %s", cfg_name)
        cfg = config[cfg_name]
        if cfg.get('modify_config', False):
            cfg = eapl_modify_config(config[cfg['ref_config']], cfg['dict_mods'])
        data = cfg['func'](data, cfg)
        if cfg.get('output_df', False):
            dfs = eapl_convert_to_list(cfg['output_df'])
            for dfname in dfs:
                eapl_kpi_logger.info("%s shape: %s", dfname, data[dfname].shape)

    eapl_kpi_logger.info("Processing End")
    return data


def eapl_kpis_write_xlsx(data, config, fname, max_rows=10000):
    # Dump the relevant EDA sheets
    sheets = []
    for cfg_key in config['config_seq']:
        sub_cfg = config[cfg_key]
        if sub_cfg.get('write_to_xlsx', False):
            sheets = sheets + eapl_convert_to_list(sub_cfg['output_df'])

    sheets = set(sheets)
    if len(sheets) > 0:
        writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        for sh_name in sheets:
            if sh_name in data.keys():
                data[sh_name][:max_rows].to_excel(writer, sheet_name=sh_name)
        writer.save()


def eapl_save_data(data, cfg):
    pkl_file = cfg['pickle_file']
    if cfg.get('pop_keys', False):
        for key in cfg['pop_keys']:
            data.pop(key, None)

    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data


def eapl_read_data(data, cfg):
    pkl_file = cfg['pickle_file']
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # data is assumed to be a dicitionary object
    pop_keys = set(cfg.get('pop_keys', []))

    for key in pop_keys:
        data.pop(key, None)

    data_keys = data.keys()
    pop2_keys = set(data_keys) - set(cfg.get('sel_keys', data_keys))
    for key in pop2_keys:
        data.pop(key, None)

    return data


def eapl_data_quality_config(data, cfg):
    dq_cfg = cfg.copy()

    df = data[dq_cfg['input_df']]
    dq_cols = list(df.columns)
    dq_cols = dq_cfg.get('sel_cols', dq_cols)
    feat_cols = sorted(list(set(dq_cols) - set(dq_cfg['gpbycols'])))
    fcol = feat_cols[0]
    ops = [['agg_feats', (fcol, 'count', 'COUNT_' + fcol)]]
    for col in feat_cols:
        if is_numeric_dtype(df[col]):
            feat_op = ['agg_feats', (col, 'nunique', 'DIST_' + col)]
        else:
            feat_op = ['agg_feats', (col, 'nunique', 'DIST_' + col)]
        ops.append(feat_op)
    dq_cfg['ops'] = ops

    data = eapl_df_agg_pct_kpis_wrapper(data, dq_cfg)

    sum_cfg = {
        'input_df': dq_cfg['output_df'],
        'output_df': dq_cfg['output_df'],
    }
    data = eapl_data_summary_wrapper(data, sum_cfg)

    return data


def eapl_gpby_correl(data, cfg):
    inp_df = data[cfg['input_df']]
    corr_method = cfg.get('corr_method', 'pearson')
    min_periods = cfg.get('min_periods', 30)
    corr_thr = cfg.get('corr_thr', 0.0)

    if cfg.get('input_df_filt', False):
        inp_df = inp_df.query(cfg['input_df_filt'])

    gpbycols = cfg.get('gpbycols', [])
    if len(gpbycols) <= 0:
        cor_df = inp_df.corr(method=corr_method, min_periods=min_periods)
    else:
        cor_df = inp_df.groupby(gpbycols).corr(method=corr_method, min_periods=min_periods)

    index_names = gpbycols + ['features']
    cor_df.index.names = index_names
    cor_df = cor_df.reset_index()

    corr_long_df = pd.melt(cor_df, id_vars=index_names, var_name='corr_feat', value_name='corr_coef')
    corr_long_df['corr_coef_abs'] = np.abs(corr_long_df['corr_coef'])
    corr_long_df = corr_long_df.query("corr_coef_abs >= @corr_thr", local_dict={'corr_thr': corr_thr})

    data[cfg['output_df']] = corr_long_df
    return data


def eapl_test_eapl_kpi():
    data = {
        'df': pd.read_csv('https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv')
    }
    cfg = {
        'func': eapl_derived_cols_wrapper,
        'input_df': 'df',
        'output_df': 'out_df',
        'ops': [
            ('df_getattr', ('rename', {'columns': {'PassengerId': 'pid'}})),
            ('df_getattr', ('xyz', {'columns': {'PassengerId': 'pid'}})),  # testing invalid operations
            ('fix_col_case', lambda x: x.lower())
        ]
    }
    data = eapl_derived_cols_wrapper(data, cfg)
    return None


if __name__ == '__main__':
    eapl_test_eapl_kpi()
