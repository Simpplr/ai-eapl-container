from .eapl_kpi import *
from .eapl_structured import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from treeinterpreter import treeinterpreter as ti
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import logging

eapl_ml_logger = logging.getLogger(__name__)


def eapl_segments_benchmarks_heatmap(data, cfg):
    all_feat_df = data[cfg['input_df']]
    seg_cols = cfg['seg_cols']
    rep_id_col = cfg['rep_id_col']
    category_col = cfg['category_col']
    agg_method = cfg.get('agg_method', 'mean')
    rank_method = cfg.get('rank_method', 'average')
    iqr_mul = cfg.get('iqr_mul', 100)
    op_args = cfg.get('ops', None)
    if cfg.get('input_df_filt', False):
        all_feat_df = all_feat_df.query(cfg['input_df_filt'])

    kpi_bm_cols = cfg['kpi_bm_cols']

    # Remove non numeric columns
    numeric_cols = [n for n, c in all_feat_df.items() if is_numeric_dtype(c)]
    id_vars = seg_cols + [rep_id_col] + [category_col]
    cols = id_vars + numeric_cols
    all_feat_df = all_feat_df[cols]
    for col in id_vars:
        all_feat_df[col] = all_feat_df[col].astype(str)
    seg_kpi_long_df = pd.melt(all_feat_df, id_vars=id_vars, var_name='kpi', value_name='kpi_value')
    seg_kpi_long_df['kpi_pct'] = seg_kpi_long_df.groupby(seg_cols + ['kpi'])['kpi_value'].rank(
        method=rank_method, pct=True, na_option='bottom')
    seg_kpi_long_df['category_pct'] = seg_kpi_long_df[category_col] + '_pct'

    hml_kpi_df = pd.pivot_table(seg_kpi_long_df, values='kpi_value', index=seg_cols + ['kpi'],
                                columns=category_col, aggfunc=agg_method).reset_index()
    hml_pct_df = pd.pivot_table(seg_kpi_long_df, values='kpi_pct', index=seg_cols + ['kpi'],
                                columns='category_pct',
                                aggfunc=agg_method).reset_index()
    hml_bm_df = hml_kpi_df.merge(hml_pct_df, how='left')
    if op_args:
        for op, args in op_args:
            hml_bm_df = eapl_derived_col_op_args(hml_bm_df, op, args)

    # Creating Rep Heat Map table
    kpi_bm_cols = list(set(kpi_bm_cols + seg_cols + ['kpi']))
    kpi_rep_cols = id_vars + ['kpi', 'kpi_value', 'kpi_pct']
    kpi_heatmap_df = seg_kpi_long_df[kpi_rep_cols].merge(hml_bm_df[kpi_bm_cols], how='left')

    data[cfg['hml_bm_df']] = hml_bm_df
    data[cfg['kpi_heatmap_df']] = kpi_heatmap_df
    return data


def eapl_rf_model(data, cfg):
    df_raw = data[cfg['train_df']].copy()
    drop_cols = cfg.get('drop_cols', None)

    train_df_filt_query = cfg.get('train_df_filt', None)
    if train_df_filt_query: df_raw.query(train_df_filt_query, inplace=True)
    eapl_ml_logger.info(f'Dataframe shape used for modeling : {df_raw.shape}')

    datepart_cols = cfg.get('datepart_cols', [])
    target_col = cfg['target_col']
    model_obj = cfg['model']
    model_trn_params = cfg['model_trn_params']
    feat_imp_df = cfg['feat_imp_df']
    oob_pred_df = cfg.get('oob_pred_df', None)
    ml_method = cfg['ml_method']
    bootstrap = cfg.get('bootstrap', True)
    oob_score = cfg.get('oob_score', True)
    n_estimators = cfg.get('n_estimators', 40)
    max_features = cfg.get('max_features', "sqrt")
    min_samples_leaf = cfg.get('min_samples_leaf', 3)
    max_n_cat = cfg.get('max_n_cat', None)
    class_weight = cfg.get('class_weight', None)
    max_depth = cfg.get('max_depth', None)
    n_rf_samples = cfg.get('n_rf_samples', None)
    n_valid = int(cfg.get('n_valid_pct', 0.2) * len(df_raw))
    verbose = cfg.get('verbose', 0)
    second_pass = cfg.get('second_pass', False)
    rf_imp_thr = cfg.get('rf_imp_thr', 0.005)
    if n_rf_samples: set_rf_samples(n_rf_samples)

    ml_func_d = {
        'RandomForestRegressor': RandomForestRegressor,
        'RandomForestClassifier': RandomForestClassifier,
    }

    model_pred_df = df_raw.reset_index().drop(columns='index')
    if drop_cols: df_raw.drop(labels=drop_cols, axis=1, inplace=True)

    for dtcol in datepart_cols:
        add_datepart(df_raw, dtcol)

    train_cats(df_raw)
    df, y, nas = proc_df(df_raw, target_col, max_n_cat=max_n_cat)

    def split_vals(a, n):
        return a[:n], a[n:]

    n_trn = len(df) - n_valid
    X_train, X_valid = split_vals(df, n_trn)
    y_train, y_valid = split_vals(y, n_trn)

    ml_func = ml_func_d[ml_method]
    m = ml_func(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf, bootstrap=bootstrap, oob_score=oob_score,
                class_weight=class_weight, verbose=verbose, n_jobs=-1)
    m.fit(X_train, y_train)

    # 2nd pass of features with only selected importance features to improve robustness
    rf_imp_cols = df.columns
    if second_pass:
        rf_imp_cols = list(
            rf_feat_importance(m, df).query("imp > @rf_imp_thr", local_dict={'rf_imp_thr': rf_imp_thr})['cols'])
        X_train, X_valid = X_train[rf_imp_cols], X_valid[rf_imp_cols]
        m.fit(X_train, y_train)

    # Dump prediction output
    model_pred_df['train_val'] = np.where(model_pred_df.index < n_trn, 'TRAIN', 'VAL')
    pred_df = pd.DataFrame(m.predict_proba(df[rf_imp_cols]))
    pred_df.columns = ['proba_' + str(c) for c in pred_df.columns]

    data[model_obj] = m
    data[model_trn_params] = (df_raw[:0].copy(), rf_imp_cols, nas, max_n_cat)
    data[feat_imp_df] = rf_feat_importance(m, df[rf_imp_cols])
    if cfg.get('model_raw_df', None):
        model_raw_df = pd.concat([model_pred_df, pred_df], axis=1)
        model_raw_df['pred_class'] = m.predict(df[rf_imp_cols])
        data[cfg['model_raw_df']] = model_raw_df

    if cfg.get('model_feat_df', None):
        model_feat_df = X_valid[rf_imp_cols]
        model_feat_df[target_col] = y_valid
        data[cfg['model_feat_df']] = model_feat_df

    m_oob_score = m.oob_score_ if hasattr(m, 'oob_score_') else 'Not Available'
    eapl_ml_logger.info(f"Model Train data score {m.score(X_train, y_train)}")
    eapl_ml_logger.info(f"OOB Score: {m_oob_score}")

    if ml_method == 'RandomForestClassifier':
        eapl_ml_logger.info(f"Validation Data Confusion Matrix")
        eapl_ml_logger.info(confusion_matrix(y_valid, m.predict(X_valid)))
        if (m.n_classes_ == 2) and (df_raw[target_col].nunique() == 2):  # Binary class problem
            tn, fp, fn, tp = confusion_matrix(y_valid, m.predict(X_valid)).ravel()
            eapl_ml_logger.info(f'Validation Accuracy: {(tn + tp) / (tn + tp + fn + fp)}, '
                                f'Sensitivity: {tp / (tp + fn)}, Specificity: {tn / (tn + fp)}')

    if n_rf_samples: reset_rf_samples()
    return data


def eapl_rf_predict(data, cfg):
    model_scoring_df = data[cfg['model_scoring_df']].copy()
    prob_bins = cfg.get('prob_bins', [0.0, 0.2499, 0.4999, 1])
    prob_labels = cfg.get('prob_labels', ['Low', 'Medium', 'High'])
    if cfg.get('model_scoring_df_filt', None): model_scoring_df.query(cfg['model_scoring_df_filt'], inplace=True)
    m = data[cfg['model']]
    trn, rf_imp_cols, nas, max_n_cat = data[cfg['model_trn_params']]
    model_output_df = cfg['model_output_df']

    model_pred_df = model_scoring_df[trn.columns]
    apply_cats(model_pred_df, trn)
    df, _, nas = proc_df(model_pred_df, max_n_cat=max_n_cat, na_dict=nas)

    if m._estimator_type == 'classifier':
        pred_df = pd.DataFrame(m.predict_proba(df[rf_imp_cols]))
        pred_df.columns = ['proba_' + str(c) for c in pred_df.columns]
        model_scoring_df['pred_class'] = m.predict(df[rf_imp_cols])
        model_scoring_df = pd.concat([model_scoring_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
                                     axis=1)
        if m.n_classes_ == 2:
            model_scoring_df['prob_category'] = pd.cut(model_scoring_df['proba_1'], bins=prob_bins,
                                                       labels=prob_labels, include_lowest=True)

    else:
        model_scoring_df['pred'] = m.predict(df[rf_imp_cols])

    if m._estimator_type == 'classifier' and m.n_classes_ == 2:
        id_cols = eapl_convert_to_list(cfg['ti_id_cols'])
        ti_output_df = cfg['ti_output_df']
        m_fval_df = model_scoring_df[id_cols + rf_imp_cols]
        m_fval_df = pd.melt(m_fval_df, id_vars=id_cols, var_name='feature', value_name='feat_value')
        pred_arr, bias_arr, cont_arr = ti.predict(m, df[rf_imp_cols])
        cont_df = pd.DataFrame(cont_arr[:, :, 1], columns=rf_imp_cols)
        cont_df = pd.concat([model_scoring_df[id_cols].reset_index(drop=True), cont_df.reset_index(drop=True)],
                            axis=1)
        cont_ldf = pd.melt(cont_df, id_vars=id_cols, var_name='feature', value_name='feat_contrib')
        cont_ldf = m_fval_df.merge(cont_ldf, how='left')
        data[ti_output_df] = cont_ldf

    data[model_output_df] = model_scoring_df
    return data


def eapl_simdeals_simscore(ldf_rdf_match, sim_feat_weights, sim_score_thr):
    sim_info = [(k + '_x', k + '_y', v) for k, v in sim_feat_weights.items()]

    ldf_rdf_match['sim_score'] = 0
    for lcol, rcol, w in sim_info:
        for col in [lcol, rcol]:
            if ldf_rdf_match[col].dtype.name == 'category':
                ldf_rdf_match[col] = ldf_rdf_match[col].astype('object')

        eval_str = f"sim_score = sim_score + ({lcol} == {rcol}) * {w}"
        ldf_rdf_match.eval(eval_str, inplace=True)

    ldf_rdf_match.query("sim_score >= @sim_score_thr", inplace=True, local_dict={'sim_score_thr': sim_score_thr})

    return ldf_rdf_match


def eapl_simdeals(data, cfg):
    sim_score_thr = cfg.get('sim_score_thr', 0.01)
    max_sim_deals = cfg.get('max_sim_deals', 2)
    sim_feat_weights = cfg['sim_feat_weights']
    sim_feat_cols = list(sim_feat_weights.keys())
    sim_reco_msg = cfg['sim_reco_msg']
    sim_reco_cols = eapl_convert_to_list(cfg['sim_reco_cols'])
    id_cols = eapl_convert_to_list(cfg['id_cols'])
    dedup_simd_cols = eapl_convert_to_list(cfg.get('dedup_simd_cols', []))
    must_match_cols = eapl_convert_to_list(cfg['must_match_cols'])
    info_cols = eapl_convert_to_list(cfg.get('info_cols', []))
    simd_cols = list(set(id_cols + must_match_cols + sim_feat_cols + info_cols + sim_reco_cols))
    sim_deals_df = cfg['sim_deals_df']
    sim_deals_msg_df = cfg['sim_deals_msg_df']
    ldf = data[cfg['left_df']]
    ldf_filt = cfg['left_df_filt']
    ldf = ldf.query(ldf_filt)

    rdf = data[cfg['right_df']]
    rdf_filt = cfg['right_df_filt']
    rdf = rdf.query(rdf_filt)

    ldf_rdf_match = ldf[simd_cols].merge(rdf[simd_cols], how='left', on=must_match_cols)
    ldf_rdf_match = eapl_simdeals_simscore(ldf_rdf_match, sim_feat_weights, sim_score_thr)

    # Drop matches with same identifiers in similar deals
    eval_str = [f"({col}_x != {col}_y)" for col in id_cols]
    eval_str = " and ".join(eval_str)
    ldf_rdf_match.query(eval_str, inplace=True)

    # Dedup multiple sim deals with common identifier
    id_cols_x = [f"{col}_x" for col in id_cols]
    for col in dedup_simd_cols:
        gpbycols = id_cols_x + [f"{col}_y"]
        ldf_rdf_match['gp_cc'] = ldf_rdf_match.groupby(gpbycols).cumcount() + 1
        ldf_rdf_match = ldf_rdf_match.query("gp_cc == 1").drop(columns='gp_cc')

    sd_cols_map = {col: f"sd_{col.rstrip('_y')}" for col in ldf_rdf_match.columns if col.endswith('_y')}
    sd_cols_map.update({f"{col}_x": col for col in id_cols})
    sd_cols = list(set(list(sd_cols_map.keys()) + ['sim_score']))
    ldf_rdf_match = ldf_rdf_match[sd_cols].rename(columns=sd_cols_map)

    ldf_rdf_match = ldf_rdf_match.sort_values(by=id_cols + ['sim_score'], ascending=False)
    ldf_rdf_match = ldf_rdf_match.groupby(id_cols).head(max_sim_deals)
    ldf_rdf_match['sim_rank'] = ldf_rdf_match.groupby(id_cols).cumcount() + 1

    sim_reco_cols = [f'sd_{col}' for col in sim_reco_cols]
    ops = [
        ['append_msgs', (sim_reco_cols, sim_reco_msg, 'sim_reco_msg')]
    ]
    ldf_rdf_match = eapl_derived_cols(ldf_rdf_match, ops)

    cfg = {
        'gpbycols': id_cols,
        'ops': [
            ['filt_topn_feats', (None, 'sim_reco_msg', 'sim_score', 'sum', 'sim_reco_msg', max_sim_deals)],
        ]
    }
    msg_df = eapl_df_agg_pct_kpis(ldf_rdf_match, cfg)

    data[sim_deals_df] = ldf_rdf_match
    data[sim_deals_msg_df] = msg_df
    return data


def eapl_topn_comb_pairs(data, cfg):
    df = data[cfg['input_df']]
    if cfg.get('input_df_filt', False):
        df = df.query(cfg['input_df_filt'])

    gpbycols = cfg.get('gpbycols', [])
    id_col = cfg['id_col']
    pair_col = cfg['pair_col']
    topn_pair_col = cfg.get('topn_pair_col', 'topn_pairs')
    nu_pairs_col = cfg.get('nu_pairs_col', 'nu_pairs')
    topn = cfg.get('topn', 3)
    topn_agg_type = cfg.get('topn_agg_type', 'count')

    gpby_id_cols = eapl_convert_to_list(gpbycols) + eapl_convert_to_list(id_col)
    df_cols = gpby_id_cols + eapl_convert_to_list(pair_col)
    df = df[df_cols].drop_duplicates()

    comb_df = df.merge(df, how='left', on=gpby_id_cols)

    scfg = {
        'gpbycols': gpbycols + [f'{pair_col}_x'],
        'ops': [
            ['filt_topn_feats',
             (f"{pair_col}_x != {pair_col}_y", f'{pair_col}_y', id_col, topn_agg_type, topn_pair_col, topn)],
            ['agg_feats', (f'{pair_col}_y', 'nunique', nu_pairs_col)],
            ['rename_cols', {f'{pair_col}_x': pair_col}],
        ]
    }
    comb_pair_df = eapl_df_agg_pct_kpis(comb_df, scfg)
    data[cfg['output_df']] = comb_pair_df
    return data


def eapl_gpby_weighted_scoring(data, cfg):
    df = data[cfg['input_df']]
    gpbycols = cfg.get('gpbycols', None)
    weights = cfg['weights']  # Dictionary containing column weights
    rank_method = cfg.get('rank_method', 'average')
    score_col = cfg.get('score_col', 'score')
    rank_col = cfg.get('rank_col', 'rank')

    wt_cols = list(weights.keys())
    df[score_col] = 0
    for col in wt_cols:
        col_pct = df.groupby(gpbycols)[col].rank(method=rank_method, na_option='bottom', pct=True)
        col_wt = weights[col]
        df[score_col] = df[score_col] + col_pct * col_wt

    df[rank_col] = df.groupby(gpbycols)[score_col].rank(method='dense', ascending=False, na_option='bottom')

    data[cfg['output_df']] = df
    return data


def eapl_seg_apriori_analysis(data, cfg):
    ap_inp_df = data[cfg['ap_inp_df']]
    gpbycols = cfg['gpbycols']
    apriori_df = None

    for seg, ap_seg_df in ap_inp_df.groupby(gpbycols):
        scfg = cfg.copy()
        data['tmp_ap_seg_df'] = ap_seg_df
        scfg['ap_inp_df'] = 'tmp_ap_seg_df'
        scfg['apriori_df'] = 'tmp_apriori_df'
        scfg['arules_df'] = None
        eapl_ml_logger.debug(f"Running combination analysis for segment : {seg}, shape: {ap_seg_df.shape}")
        data = eapl_apriori_analysis(data, scfg)
        tmp_apriori_df = data['tmp_apriori_df']
        seg = eapl_convert_to_list(seg)
        for idx, col in enumerate(gpbycols):
            tmp_apriori_df.insert(idx, col, seg[idx])
        eapl_ml_logger.debug(f"# of combinations for segment : {seg}, shape: {tmp_apriori_df.shape}")

        if tmp_apriori_df.shape[0] > 0:
            apriori_df = pd.concat([apriori_df, tmp_apriori_df], axis=0)

    data[cfg['apriori_df']] = apriori_df

    if 'tmp_apriori_df' in data:
        del data['tmp_apriori_df']

    return data


def eapl_apriori_analysis(data, cfg):
    ap_inp_df = data[cfg['ap_inp_df']].copy()
    ap_inp_df_filt = cfg.get('ap_inp_df_filt', None)
    ap_cols = cfg['ap_cols']
    arules_df_n = cfg.get('arules_df', None)
    min_support = cfg.get('min_support', 0.01)
    min_support_count = cfg.get('min_support_count', 5)
    metric = cfg.get('metric', "lift")
    min_threshold = cfg.get('min_threshold', 1.0)
    score_calc = cfg.get('score_calc', "comb_len * count")

    if ap_inp_df_filt:
        ap_inp_df.query(ap_inp_df_filt, inplace=True)

    # Convert categorical columns to object type
    for col in ap_cols:
        if ap_inp_df[col].dtype.name == 'category':
            ap_inp_df[col] = ap_inp_df[col].astype('object')

    # Remove records with missing values
    qfilt = [f"({col} == {col})" for col in ap_cols]
    qfilt = " and ".join(qfilt)
    ap_inp_df.query(qfilt, inplace=True)

    for col in ap_cols:
        ap_inp_df[col] = ap_inp_df[col].apply(lambda x: eapl_convert_to_list(x))

    txn_list = ap_inp_df[ap_cols].values.tolist()

    txn_feats = []
    for le in txn_list:
        txn_feats.append(sum(le, []))

    te = TransactionEncoder()
    te_ary = te.fit(txn_feats).transform(txn_feats)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    if min_support > 1.0:
        min_support = min_support / df.shape[0]
    min_support_count_pct = min_support_count / df.shape[0]
    min_support = max(min_support, min_support_count_pct)

    apriori_df = apriori(df, min_support=min_support, use_colnames=True)
    apriori_df_valid = apriori_df.shape[0] >= 1

    if arules_df_n and apriori_df_valid:
        arules_df = association_rules(apriori_df, metric=metric, min_threshold=min_threshold)
        arules_df.sort_values(by=metric, ascending=False, inplace=True)
        arules_df['antece_len'] = arules_df['antecedents'].str.len()
        arules_df['conseq_len'] = arules_df['consequents'].str.len()
        arules_df.eval("ac_len = antece_len + conseq_len")
        data[arules_df_n] = arules_df

    if apriori_df_valid:
        apriori_df['comb_len'] = apriori_df['itemsets'].str.len()
        apriori_df['total_count'] = len(txn_feats)
        apriori_df['count'] = apriori_df.eval("support * total_count")
        apriori_df['itemsets'] = apriori_df['itemsets'].apply(list)
        apriori_df['comb_score'] = apriori_df.eval(score_calc)
        apriori_df = apriori_df.sort_values(by=['comb_score', 'comb_len'],
                                            ascending=[False, False])
        apriori_df['comb_rank'] = apriori_df['comb_score'].expanding().count()

    data[cfg['apriori_df']] = apriori_df

    return data


def eapl_kpi_benchmarking(data, cfg):
    train_df = data[cfg['train_df']]
    test_df = data[cfg['test_df']]
    id_col = cfg['id_col']
    drop_cols_g = eapl_convert_to_list(cfg.get('drop_cols', []))
    bm_params = cfg['bm_params']

    # Iterative benchmarking methodology
    bm_final_df = None
    bm_rep_list = []
    for bm_params_dict in bm_params:
        seg_cols = bm_params_dict['seg_cols']
        qt_col, qt_min, qt_max = bm_params_dict['qt_params']
        att_col, att_min, att_max = bm_params_dict['att_params']
        min_bm_thr = bm_params_dict.get('min_bm_thr', 1)

        att_train_df = train_df.query(f'{att_col} >= {att_min} and {att_col} <= {att_max}')
        cy_df_cols = [id_col, qt_col] + eapl_convert_to_list(seg_cols)
        cy_df = test_df[cy_df_cols].query(f'{id_col} != {bm_rep_list}')
        cy_df.eval(f"{qt_col}_min = {qt_col} * {qt_min}", inplace=True)
        cy_df.eval(f"{qt_col}_max = {qt_col} * {qt_max}", inplace=True)

        drop_cols = set(drop_cols_g + [att_col])
        ly_df = att_train_df.drop(labels=drop_cols, axis=1)
        ly_df_map = {col: f'{col}_ly' for col in [id_col, qt_col]}
        ly_df = ly_df.rename(columns=ly_df_map)

        num_reps = cy_df.shape[0]
        step_size = 50
        drop_cols = [f'{qt_col}', f'{qt_col}_min', f'{qt_col}_max', f'{id_col}_ly', f'{qt_col}_ly']

        cy_df = cy_df.loc[:, ~cy_df.columns.duplicated()]
        ly_df = ly_df.loc[:, ~ly_df.columns.duplicated()]
        bm_df = None
        for i in range(0, num_reps, step_size):
            cy_i_df = cy_df.iloc[i:i + step_size, :].merge(ly_df, how='inner', on=seg_cols)
            cy_i_df = cy_i_df.query(f"{qt_col}_min <= {qt_col}_ly <= {qt_col}_max")
            cy_i_df = cy_i_df.drop(labels=drop_cols, axis=1)

            bm_i_df = cy_i_df.groupby(id_col, as_index=False).mean()
            cy_cnt_df = cy_i_df.groupby(id_col, as_index=False).agg({seg_cols[0]: 'count'}).rename(
                columns={seg_cols[0]: 'count'})
            bm_i_df = bm_i_df.merge(cy_cnt_df, how='left', on=id_col)
            bm_df = pd.concat([bm_df, bm_i_df], axis=0, sort=True)
        ben_cols_map = {col: (col + '_bm') for col in bm_df.columns if col not in [id_col, 'count']}

        bm_df = bm_df.rename(columns=ben_cols_map)
        bm_df = bm_df.query(f'count > {min_bm_thr}')

        if bm_final_df is None:
            bm_final_df = bm_df
        else:
            bm_final_df = bm_final_df.append(bm_df)
        bm_rep_list = list(bm_final_df[id_col].unique())
        eapl_ml_logger.debug(f"Benchmarking using seg columns: {seg_cols}")
        eapl_ml_logger.debug(f"# of Reps: {bm_df.shape[0]} meeting threshold count of {min_bm_thr}")
        eapl_ml_logger.debug(f"Total # of Reps with benchmark available: {bm_final_df.shape[0]}")

    data[cfg['output_df']] = bm_final_df
    return data
