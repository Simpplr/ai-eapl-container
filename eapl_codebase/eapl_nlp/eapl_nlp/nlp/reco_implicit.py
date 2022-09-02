import implicit
from scipy import sparse
import pandas as pd
import numpy as np
from eapl.df.eapl_kpi import eapl_convert_to_list

try:
    from .nlp_glob import nlp_func_map, nlp_glob
    from .nlp_utils import nlp_utils_fmap
    from .eapl_kpi_ref import eapl_kpi_fmap
except ImportError:
    from nlp_glob import nlp_func_map, nlp_glob
    from nlp_utils import nlp_utils_fmap
    from eapl_kpi_ref import eapl_kpi_fmap

_implicit_algo_methods = {
    'als': implicit.als.AlternatingLeastSquares,
    'bpr': implicit.bpr.BayesianPersonalizedRanking,
    'lmf': implicit.lmf.LogisticMatrixFactorization,
    'nms': implicit.approximate_als.NMSLibAlternatingLeastSquares,
    'annoy': implicit.approximate_als.AnnoyAlternatingLeastSquares,
    'faiss': implicit.approximate_als.FaissAlternatingLeastSquares
}


def eapl_implicit_data_prep(data, cfg):
    user_item_df_key = cfg.get("user_item_df_key", "user_item_df")
    user_id = cfg.get("user_id_col", "user_id")
    item_id = cfg.get("item_id_col", "item_id")
    events = cfg.get("events", "event_col")  # TBD: Choose appropriate convention
    sprs_matx_test_key = cfg.get("sprs_matx_test_key", "sprs_matx_test")
    ui_mapping_key = cfg.get("ui_mapping_key", "ui_mapping_dct")
    map_user_id = cfg.get("map_user_id", "map_uid2idx_df")
    map_item_id = cfg.get("map_item_id", "map_iid2idx_df")

    df = data[user_item_df_key].copy()
    map_uid2idx_df = pd.DataFrame()
    map_iid2idx_df = pd.DataFrame()
    df[user_id] = map_uid2idx_df[user_id] = df[user_id].astype('category')
    df[item_id] = map_iid2idx_df[item_id] = df[item_id].astype('category')
    df[user_id] = map_uid2idx_df[f"{user_id}_uid2idx"] = df[user_id].cat.codes
    df[item_id] = map_iid2idx_df[f"{item_id}_iid2idx"] = df[item_id].cat.codes
    map_uid2idx_df = map_uid2idx_df.drop_duplicates()
    map_iid2idx_df = map_iid2idx_df.drop_duplicates()

    if events not in df.columns:
        df[events] = 1

    if not df[item_id].empty:
        sprs_matx_test = sparse.csr_matrix((df[events].astype(np.float), (df[user_id], df[item_id])))
        ui_mapping_dct = {map_user_id: map_uid2idx_df, map_item_id: map_iid2idx_df}

        data[ui_mapping_key] = ui_mapping_dct
        data[sprs_matx_test_key] = sprs_matx_test  # predicting

    return data


def eapl_implicit_init(data, cfg):
    model_params = cfg.get('model_params', {}).copy()
    data_model_params_key = cfg.get('data_model_params_key', None)
    algo = cfg['algo']
    model_key = cfg["model_key"]

    if data_model_params_key:
        data_model_params = data[data_model_params_key]
        model_params.update(data_model_params)

    if algo in _implicit_algo_methods:
        implicit_func = _implicit_algo_methods[algo]
        model = implicit_func(**model_params)

    data[model_key] = model

    return data


def eapl_implicit_train_model(data, cfg):
    model_key = cfg["model_key"]
    sprs_matx_train_key = cfg.get("sprs_matx_test_key", "sprs_matx_test")
    alpha_val = cfg.get("alpha_val", 1)  # Metric that indicates if the user likes the item

    if sprs_matx_train_key in data.keys():
        train_data = data[sprs_matx_train_key]
        model = data[model_key]

        train_data = train_data * alpha_val
        model.fit(train_data)

        data[model_key] = model

    return data


def eapl_implicit_recommend_user(data, cfg):
    model_key = cfg["model_key"]
    model = data[model_key]
    top_n = cfg.get("top_n", 10)
    reco_user_id_key = cfg.get("reco_user_id_key", "reco_user_id_key")
    reco_user_ids = data.get(reco_user_id_key, None)
    filter_already_liked_items = cfg.get("liked_item", True)
    sprs_matx_test_key = cfg.get("sprs_matx_test_key", "sprs_matx_test")
    sparse_user_item = data[sprs_matx_test_key]
    out_key = cfg.get("user_reco_out_key", "user_recommendations")
    reco_type = cfg.get("reco_type", "recommend")
    item_key = cfg.get("item_key", "item_iid2idx")
    user_key = cfg.get("user_key", "user_uid2idx")
    score_key = cfg.get("score_key", "score")
    model_score_threshold = cfg.get("model_score_threshold", -1)
    reco_df = pd.DataFrame()
    if reco_type == "recommend_all":
        reco_user_ids = np.arange(sparse_user_item.shape[0])
    user_recommendations = model.recommend(userid=reco_user_ids, user_items=sparse_user_item, N=top_n,
                                           filter_already_liked_items=filter_already_liked_items)
    reco_df['recommendations'] = pd.DataFrame(np.dstack(user_recommendations).tolist()).apply(lambda x: list(x), axis=1)
    reco_df[user_key] = reco_user_ids
    reco_df = reco_df.explode('recommendations').reset_index(drop=True)
    reco_df = pd.concat(
        [pd.DataFrame(reco_df["recommendations"].to_list(), columns=[item_key, score_key]), reco_df[user_key]], axis=1)
    reco_df[score_key] = pd.to_numeric(reco_df[score_key])
    reco_df = reco_df.query(f'not({score_key} == 0 and {item_key}==0) and {score_key} > {model_score_threshold}')
    reco_df[score_key] = reco_df.groupby(user_key)[score_key].rank(ascending=True, na_option='bottom', pct=True)
    data[out_key] = reco_df
    return data


collab_implicit_fmap = {
    "eapl_implicit_data_prep": eapl_implicit_data_prep,
    "eapl_implicit_init": eapl_implicit_init,
    "eapl_implicit_train_model": eapl_implicit_train_model,
    "eapl_implicit_recommend_user": eapl_implicit_recommend_user

}
nlp_func_map.update(collab_implicit_fmap)


# TBD : Needs to updated for changes and tested
def colab_filt_init_setup():
    # Test config needs to be added as en enhancement.
    return


if __name__ == "__main__":
    colab_filt_init_setup()
