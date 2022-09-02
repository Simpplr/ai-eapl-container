import pandas as pd
import numpy as np
from datetime import datetime as dt
from freezegun import freeze_time
import logging

from .nlp_glob import *
from .nlp_ops import *
from .elastic_search_custom import *
from .reco_implicit import *
from .simpplr_mongodb_connection import *

nlp_tags_logger = logging.getLogger(__name__)


def raise_key_error(lista, listb):
    for req_key in lista:
        if req_key not in listb:
            raise KeyError(f"HTTP Error 400: Keynotfound, Missing {req_key}")


def simpplr_data_harmonisation(data, cfg):
    """A function to check if all the tables are in place and contain the columns required for the operation.
       Generates people followed and site followed tables from the raw tables

    :param data: input data containing all the tables loaded from MongoDB
    :type data: dict

    :param cfg: input configuration containing the operation to be performed, containing the following keys:
        - func: Contains the target function name which needs to be called i.e. `simpplr_data_harmonisation`

    :returns data: output tables with the people and content columns mapped to their respective ids
    :rtype: dict
    """
    people_df = data['people_raw_df']
    people_followed_df = data['people_followed_raw_df']
    site_df = data['site_raw_df']
    site_role_df = data['site_role_df']
    site_followed_key = cfg.get('site_followed_key', 'group_followed_df')
    people_followed_key = cfg.get('people_followed_key', 'people_followed_df')

    people_mapping_dict = pd.Series(people_df.Simpplr__People__Id.values, index=people_df.Simpplr__User__c).to_dict()
    people_followed_df['ParentId'] = people_followed_df.ParentId.map(people_mapping_dict)
    people_followed_df['Simpplr__People__Id'] = people_followed_df.SubscriberId.map(people_mapping_dict)
    people_followed_df = people_followed_df[['ParentId', 'Simpplr__People__Id']]
    people_followed_df = people_followed_df.fillna(' ')

    site_role_df = site_role_df.rename(columns={"Simpplr__Site__c": "CollaborationGroupId",
                                                "Simpplr__People__c": "Simpplr__People__Id",
                                                "Simpplr__Is_Member__c": "member"})
    site_role_df['member'] = site_role_df['member'].map({"true": 1, "false": 0})
    site_role_df['member'].fillna(0, inplace=True)
    site_role_df['follower'] = 1 - site_role_df['member']

    # Create Site Data Table
    site_df = site_df.rename(columns={"Id": "Simpplr__Site__r_Id"})

    # Carousel Content cleanup
    cont_id = 'Simpplr__Simpplr_Content__c'
    car_df = data['carousel_content_df']
    car_df = car_df.drop_duplicates(subset=cont_id).query(f"({cont_id} != 'None') and ({cont_id} == {cont_id})")
    data['carousel_content_df'] = car_df

    data['site_raw_df'] = site_df[['Simpplr__Site__r_Id', 'Simpplr__Site_Type__c']]
    data[people_followed_key] = people_followed_df.drop_duplicates()
    data[site_followed_key] = site_role_df.drop_duplicates(subset=['CollaborationGroupId', 'Simpplr__People__Id'])
    return data


def simpplr_page_reco_data_prep(data, cfg):
    """A function that prepares the data for the page recommendation operation

    :param data: input data containing all the tables loaded from MongoDB
    :param cfg: configuration containing the operation to be performed, containing the following keys
        - :func str: Contains the target function name which needs to be called i.e. `simpplr_page_reco_data_prep`
        - :content_id_key str: Identifier for content id in the input data
        - :people_id_key str: Identifier for user id column name in the input data
        - :content_data_key str: Contains the key for the content table dataframe present in the data dict
        - :user_item_data_key str: Contains the key for the user-content interaction table dataframe present in the data dict
        - :min_user_content int: Contains the minimum number of content items to be considered for the user
        - :out_key str: Contains the key for the filtered content interaction table for collaborative filtering in the data dict

    :return: data dictionary containing the tables loaded from MongoDB
    """
    content_id_key = cfg.get("content_id_key", "Simpplr__Content__Id")
    people_id_key = cfg.get("people_id_key", "Simpplr__People__Id")
    user_item_data_key = cfg.get("user_item_data_key", "item_raw_df")
    content_data_key = cfg.get("content_data_key", "content_raw_df")
    min_user_content = cfg.get("min_user_content", 5)
    out_key = cfg.get("out_key", "user_item_raw_df")

    uid_cid_list = [people_id_key, content_id_key]
    user_item_data = data[user_item_data_key][uid_cid_list]
    content_data = data[content_data_key][[content_id_key, "Simpplr__Is_Published__c"]]

    user_item_content_data = user_item_data.merge(content_data, on=content_id_key, how='left')
    user_item_content_data = user_item_content_data.query("Simpplr__Is_Published__c == 'true'")
    user_item_content_unq_data = user_item_content_data.drop_duplicates(subset=uid_cid_list).reset_index(drop=True)
    ui_count_df = user_item_content_unq_data.groupby(people_id_key, as_index=False).size()
    ui_count_df = ui_count_df.rename(columns={'size': 'ui_count'})
    user_item_filtered_data = user_item_content_unq_data.groupby(people_id_key).filter(
        lambda x: len(x) > min_user_content)

    data['user_item_unq_df'] = user_item_content_unq_data[uid_cid_list]
    data['ui_count_df'] = ui_count_df
    data.pop('item_raw_df')
    data[out_key] = user_item_filtered_data
    return data


def simpplr_gen_reco(data, cfg):
    """Generates recommendations for the given user/users using the given user/users

    :param data: Contains the data for accessing the tables and sparse matrices
    :type data: dict

    :param cfg: Contains the following keys which are checked for validity
        - :model_key str: Name of the key in the data dict that contains the model
        - :index str: Name of the Document DB index from where the data needs to be fetched
        - :user_id_col str: Name of the column in the indexed content that contains the user id
        - :item_id_col str: Name of the column in the indexed content that contains the item id
        - :sprs_matx_test_key str: Name of the key in the data dict that contains the sparse matrix of the interaction table
        - :top_n int: Number of recommendations to be generated
        - :min_user_content int: Minimum number of content items a user should have to be considered
        - :days_to_subtract int: Number of days to be subtracted from the current date to get the date range
        - :days_to_subtract_pf_sf int: Number of days to be subtracted from the current date to get the date range for the people and site followed based recommendations
        - :model_score_threshold float: Minimum score for the model to be considered while generating the recommendations for the user
        - :out_key str: Name of the key in the data dict in which recommendations should be returned
    :type cfg: dict

    :return data: Returns the data dict with the recommendations for the user/users
    """

    user_reco_out_key = cfg.get("user_reco_out_key", "user_recommendations")
    user_id_col = cfg.get("user_id_col", "user_id")
    item_id_col = cfg.get("item_id_col", "item_id")
    ui_mapping_key = cfg.get("ui_mapping_key", "ui_mapping_dct")
    score_key = cfg.get("score_key", "score")
    top_n = cfg.get("top_n", 50)
    min_user_content = cfg.get("min_user_content", 5),
    days_to_subtract = cfg.get("days_to_subtract", 30)
    days_to_subtract_pf_sf = cfg.get("days_to_subtract_pf_sf", 8)
    model_score_threshold = cfg.get("model_score_threshold", 0)
    user_recos = []
    out_key = cfg.get("out_key", "user_reco")
    index = cfg["index"]
    content_raw_df = data['content_raw_df']
    test_freeze_time = data.get("test_freeze_time", None)

    # Create reference table of User IDs with unique content interaction Count
    ui_count_df = data['ui_count_df']
    user_item_unq_df = data['user_item_unq_df']
    user_item_unq_df["viewed"] = 1
    people_df = data['people_raw_df'][[user_id_col]]
    cold_start_user_dct = {user_id_col: 'cold_start_def_user'}
    people_df = people_df.append(cold_start_user_dct, ignore_index=True)
    people_df = people_df.merge(ui_count_df, on=user_id_col, how='left')
    people_df['ui_count'] = people_df['ui_count'].fillna(0)

    # Tables for people and site followed
    site_followed_df = data['group_followed_df']
    site_followed_df['site_followed'] = 1
    people_followed_df = data['people_followed_df']
    people_followed_df['people_followed'] = 1

    # Table for Must Read and Carousel contents
    carousel_content_df = data['carousel_content_df']
    carousel_content_df = carousel_content_df.rename(columns={'Simpplr__Simpplr_Content__c': item_id_col})
    carousel_content_df = carousel_content_df[[item_id_col]]

    must_read_audit_df = data['must_read_audit_df']
    for dt_col in ["Simpplr__Expiry_DateTime__c", "Simpplr__Mark_DateTime__c", "Simpplr__Removed_DateTime__c"]:
        must_read_audit_df[dt_col] = pd.to_datetime(must_read_audit_df[dt_col], utc=True, errors='coerce')
    must_read_conf_hist_df = data['must_read_conf_hist_df']

    must_read_audit_df = must_read_audit_df.rename(columns={'Simpplr__Content__c': item_id_col})
    must_read_conf_hist_df = must_read_conf_hist_df.rename(columns={'Simpplr__Content__c': item_id_col,
                                                                    'Simpplr__People__c': user_id_col})

    if test_freeze_time:
        freezer = freeze_time(test_freeze_time)
        freezer.start()

    # Filtering parameters setup
    current_datetime = dt.now()
    current_datetime = current_datetime.replace(tzinfo=timezone.utc)
    offset_date = current_datetime - datetime.timedelta(days=days_to_subtract)
    offset_date_sf_pf = current_datetime - datetime.timedelta(days=days_to_subtract_pf_sf)

    if test_freeze_time:
        freezer.stop()

    publish_start_date_key = 'Simpplr__Publish_Start_DateTime__c'
    content_pub_key = "Simpplr__Is_Published__c"
    content_deleted_key = "Simpplr__Is_Deleted__c"
    content_type = "Simpplr__Type__c"
    content_type_lc_strip = "Simpplr__Type__lc_strip__c"
    event_expiry_key = "Simpplr__Event_End_DateTime__c"
    page_category = "Simpplr__Pages_Category__r_Simpplr__Name__c"
    popularity_score = "Simpplr__Popularity_Score__c"
    author_id = "Simpplr__Primary_Author__r_Id"
    site_id = "Simpplr__Site__r_Id"
    site_type = "Simpplr__Site_Type__c"
    aud_type = "Simpplr__Audience_Type__c"
    expiry_dt = "Simpplr__Expiry_DateTime__c"
    ma_mark_dt = "Simpplr__Mark_DateTime__c"
    ma_rem_dt = "Simpplr__Removed_DateTime__c"

    # Data Type mapping
    content_raw_df[publish_start_date_key] = pd.to_datetime(content_raw_df[publish_start_date_key], utc=True,
                                                            errors='coerce')
    content_raw_df[event_expiry_key] = pd.to_datetime(content_raw_df[event_expiry_key], utc=True, errors='coerce')
    content_raw_df[event_expiry_key] = content_raw_df[event_expiry_key].fillna(current_datetime)
    content_raw_df[popularity_score] = pd.to_numeric(content_raw_df[popularity_score], errors='coerce')
    content_raw_df[page_category] = content_raw_df[page_category].str.strip().str.lower()
    content_raw_df[content_type_lc_strip] = content_raw_df[content_type].str.strip().str.lower()

    def filter_and_rank(user_reco_df, content_raw_df):
        df = user_reco_df.merge(content_raw_df, on=item_id_col, how="left")
        cur_datetime = current_datetime
        off_date = offset_date
        filt_df = df.query(f"(({content_deleted_key} != 'true') and "
                           f"(({content_type_lc_strip} == 'event' and {event_expiry_key} > @cur_datetime) or "
                           f"({content_type_lc_strip} != 'event' and "
                           f"(({publish_start_date_key} > @off_date) or (reco_method == 'Must Read Reco')))) and "
                           f"({score_key} > {model_score_threshold}))"
                           )

        # Map site followed and People followed flags
        filt_df = filt_df.merge(people_followed_df, left_on=[user_id_col, author_id],
                                right_on=[user_id_col, 'ParentId'], how='left')
        filt_df = filt_df.merge(site_followed_df, left_on=[user_id_col, site_id],
                                right_on=[user_id_col, 'CollaborationGroupId'], how='left')
        filt_df['site_followed'].fillna(0, inplace=True)
        filt_df['people_followed'].fillna(0, inplace=True)

        # Weighted average based ranking of recommendations
        filt_groupby_df = filt_df.groupby(user_id_col)
        filt_df['recency_score'] = filt_groupby_df[publish_start_date_key].rank(
            ascending=True, na_option='bottom', pct=True)
        filt_df['popl_score'] = filt_groupby_df[popularity_score].rank(
            ascending=True, na_option='bottom', pct=True)
        filt_df['wt_score'] = filt_df.eval(
            f"0.3*{score_key} + 0.2*site_followed + 0.2*people_followed + 0.15*recency_score + 0.15*popl_score")
        filt_df = filt_df.sort_values(by=[user_id_col, 'wt_score'], ascending=[True, False])
        filt_df = filt_df.drop_duplicates(subset=[user_id_col, item_id_col])

        # Filter recommendations not applicable to specific users based on rules
        filt_df = filt_df.merge(data['site_raw_df'], on=site_id, how='left')
        filt_df = filt_df.query(f"{content_deleted_key} == 'false' and {content_pub_key} == 'true' and "
                                f"(site_followed > 0 or {site_type} == 'Public' or "
                                f"({content_type_lc_strip} == 'blogpost' and people_followed > 0))")
        return filt_df

    user_reco_df = pd.DataFrame()
    if ui_mapping_key in data.keys():
        # user and content id mapping used for collaborative filtering
        map_uid2idx_df = data[ui_mapping_key]['map_uid2idx_df']
        map_iid2idx_df = data[ui_mapping_key]['map_iid2idx_df']
        # Generate recos from collaborative filtering
        cfg_reco = cfg.copy()
        cfg_reco["item_key"] = f"{item_id_col}_iid2idx"
        cfg_reco["score_key"] = score_key
        cfg_reco["user_key"] = f"{user_id_col}_uid2idx"
        cfg_reco["reco_type"] = "recommend_all"
        data = eapl_implicit_recommend_user(data, cfg_reco)
        user_reco_df = data[user_reco_out_key]
        user_reco_df = user_reco_df.merge(map_uid2idx_df, on=f"{user_id_col}_uid2idx", how='left')
        user_reco_df = user_reco_df.merge(map_iid2idx_df, on=f"{item_id_col}_iid2idx", how='left')
        user_reco_df = user_reco_df[[user_id_col, item_id_col, score_key]]
        user_reco_df['reco_method'] = 'Collaborative Filtering'
        user_reco_df['message'] = 'Recommendations from AI Model'

    # Content Reco from Must Read content
    must_read_reco_df = people_df.copy()
    must_read_reco_df["join_flag"] = 1
    must_read_audit_df = must_read_audit_df.query(
        f"(({expiry_dt} > @current_datetime) or ({expiry_dt} != {expiry_dt})) and {ma_rem_dt} != {ma_rem_dt}")
    must_read_audit_df = must_read_audit_df.sort_values(by=[item_id_col, ma_mark_dt], ascending=[True, False])
    must_read_audit_df = must_read_audit_df.drop_duplicates(subset=[item_id_col])
    must_read_audit_df["join_flag"] = 1
    must_read_reco_df = must_read_reco_df.merge(must_read_audit_df, on="join_flag", how="left")
    content_site_df = content_raw_df[[item_id_col, site_id]]
    must_read_reco_df = must_read_reco_df.merge(content_site_df, on=item_id_col, how='left')
    must_read_reco_df = must_read_reco_df.merge(site_followed_df, left_on=[user_id_col, site_id],
                                                right_on=[user_id_col, 'CollaborationGroupId'], how='left')
    # Possible values: "site_members_and_followers"/"site_members"/"everyone"
    must_read_reco_df = must_read_reco_df.query(
        f"({aud_type} == 'everyone') or "
        f"({aud_type} == 'site_members' and member == 1) or "
        f"({aud_type} == 'site_members_and_followers' and (member == 1 or follower == 1))")
    must_read_conf_hist_df["must_read_flag"] = 1
    must_read_reco_df = must_read_reco_df.merge(must_read_conf_hist_df, on=[user_id_col, item_id_col], how='left')
    must_read_reco_df = must_read_reco_df.query("must_read_flag != 1")[[user_id_col, item_id_col]]
    must_read_reco_df[score_key], must_read_reco_df["reco_method"] = 4, "Must Read Reco"
    must_read_reco_df['message'] = 'Recommendations from AI Model'

    # Content Reco from Carousel content
    car_reco_df = people_df.copy()
    car_reco_df["join_flag"] = 1
    carousel_content_df["join_flag"] = 1
    carousel_df = pd.DataFrame()
    chunk_size = int(len(car_reco_df) * len(carousel_content_df) / 500000)
    chunk_size = chunk_size if chunk_size > 1 else 1
    for chunked_df_ in np.array_split(car_reco_df, chunk_size):
        chunked_df = chunked_df_.merge(carousel_content_df, on="join_flag", how="left")
        chunked_df[score_key], chunked_df["reco_method"] = 2, "Carousel Reco"
        chunked_df = filter_and_rank(chunked_df, content_raw_df)
        chunked_df = chunked_df.merge(user_item_unq_df, on=[user_id_col, item_id_col], how='left')
        chunked_df = chunked_df.query("viewed != 1")[[item_id_col, score_key, "reco_method", user_id_col]]
        chunked_df = chunked_df.groupby([user_id_col]).head(top_n)
        chunked_df['message'] = 'Recommendations from AI Model'
        carousel_df = pd.concat([carousel_df, chunked_df])

    # Content Reco from Site Followed and People followed
    pf_sf_cont_df = content_raw_df.query(f"{publish_start_date_key} >= @offset_date_sf_pf")
    pf_sf_cont_df = pf_sf_cont_df.sort_values(by=[publish_start_date_key, popularity_score],
                                              ascending=[False, False]).head(top_n)
    pf_sf_cont_df["join_flag"] = 1
    pf_sf_user_df = people_df.copy()
    pf_sf_user_df["join_flag"] = 1
    pf_sf_user_df = pf_sf_user_df.merge(pf_sf_cont_df, on="join_flag", how='left')
    pf_sf_user_df = pf_sf_user_df[[user_id_col, item_id_col, author_id, site_id]]
    pf_sf_user_df = pf_sf_user_df.merge(people_followed_df, left_on=[user_id_col, author_id],
                                        right_on=[user_id_col, 'ParentId'], how='left')
    pf_sf_user_df = pf_sf_user_df.merge(site_followed_df, left_on=[user_id_col, site_id],
                                        right_on=[user_id_col, 'CollaborationGroupId'], how='left')
    pf_sf_user_df = pf_sf_user_df.query("site_followed == 1 or people_followed == 1")
    pf_sf_user_df = pf_sf_user_df.merge(user_item_unq_df, on=[user_id_col, item_id_col], how='left')
    pf_sf_user_df = pf_sf_user_df.query("viewed != 1")[[user_id_col, item_id_col]]
    pf_sf_user_df[score_key], pf_sf_user_df["reco_method"] = 1, "People/Site Followed"
    pf_sf_user_df['message'] = 'Recommendations from AI Model'

    # Generate fallback recommendations based on recency and popularity
    fallback_cont_df = content_raw_df.query(f"{publish_start_date_key} >= @offset_date")
    fallback_cont_df = fallback_cont_df.sort_values(by=[publish_start_date_key, popularity_score],
                                                    ascending=[False, False]).head(top_n)
    fallback_cont_df["join_flag"] = 1
    fallback_user_df = people_df.query(f"ui_count <= @min_user_content")
    fallback_user_df["join_flag"] = 1
    fallback_user_df = fallback_user_df.merge(fallback_cont_df, on="join_flag", how='left')
    fallback_user_df = fallback_user_df[[user_id_col, item_id_col]]
    fallback_user_df = fallback_user_df.merge(user_item_unq_df, on=[user_id_col, item_id_col], how='left')
    fallback_user_df = fallback_user_df.query("viewed != 1")[[user_id_col, item_id_col]]
    fallback_user_df[score_key], fallback_user_df["reco_method"] = 1, "Cold Start"
    fallback_user_df['message'] = 'Recommendations from Cold Start'

    # Combine multiple approaches recommendations and filter based on business rules
    filt_df = filter_and_rank(must_read_reco_df, content_raw_df)
    del must_read_reco_df
    filt_df = pd.concat([filt_df, filter_and_rank(carousel_df, content_raw_df)])
    del carousel_df
    filt_df = pd.concat([filt_df, filter_and_rank(pf_sf_user_df, content_raw_df)])
    del pf_sf_user_df
    filt_df = pd.concat([filt_df, filter_and_rank(user_reco_df, content_raw_df)])
    filt_df = pd.concat([filt_df, filter_and_rank(fallback_user_df, content_raw_df)])

    # Restrict the number of recos per page category to 2
    filt_df = filt_df.groupby([user_id_col, page_category]).head(2)
    filt_df = filt_df.groupby([user_id_col]).head(top_n)
    # Convert recommendations for indexing
    if filt_df.shape[0] > 0:
        filt_df['recommendations'] = filt_df.apply(lambda x: {
            item_id_col: x[item_id_col], score_key: x['wt_score'], "message": x['message'],
            "reco_method": x['reco_method']}, axis=1)
    else:
        filt_df['recommendations'] = filt_df.apply(lambda x: {}, axis=1)
    collab_reco_df = filt_df.groupby(user_id_col)['recommendations'].apply(list).to_frame().reset_index()

    # TODO: Added just for sanity checks. Can be deleted
    sf_df = site_followed_df.groupby(user_id_col)['CollaborationGroupId'].apply(list).to_frame().reset_index()
    sf_df = sf_df.rename(columns={'CollaborationGroupId': 'site_followed'})

    # People followed
    pf_df = people_followed_df.groupby(user_id_col)['ParentId'].apply(list).to_frame().reset_index()
    pf_df = pf_df.rename(columns={'ParentId': 'people_followed'})

    # Combine data
    reco_df = people_df.merge(collab_reco_df, on=user_id_col, how='left')
    reco_df = reco_df.merge(sf_df, on=user_id_col, how='left')
    reco_df = reco_df.merge(pf_df, on=user_id_col, how='left')

    # Fill nans specific for each column
    reco_df = reco_df.fillna("FILL_ELIST")
    for col in ['recommendations', 'site_followed', 'people_followed']:
        reco_df[col] = reco_df[col].apply(lambda x: [] if x == 'FILL_ELIST' else x)

    # Transform the output to response format
    reco_df = reco_df.rename(columns={user_id_col: "reco_user_id"})
    extra_fields = {"_op_type": "index", "_index": index, 'record_status': 'active'}
    reco_df = reco_df.assign(**extra_fields)
    reco_df["_id"] = reco_df["reco_user_id"]
    user_recos = reco_df.to_dict(orient='records')

    data[out_key] = user_recos
    return data


def simpplr_get_reco(data, cfg):
    """This function is used to get the realtime recommendations for the users

    :param data: input data dict containing the reference to objects to generate the recommendations
    :param cfg: configuration for the model containing the user ids for whom we need to get the recommendations
        - :index str: Name of the ES index from which the recommendations needs to be retrieved
        - :top_recom int: Number of recommendations to be retrieved
        - :user_ids Any[List, String]: List of user ids for whom the recommendations needs to be generated
        - :reco_out_key str: Key in the `data` that needs to be populated with the recommendations

    :return data: Returns data with the recommendations populated in the `reco_out_key`

    """

    es_obj = cfg.get('es_obj', 'es_obj')
    es = data[es_obj]
    top_recom = cfg.get("top_recom", 50)
    es_index = cfg["index"]
    reco_out_key = cfg.get('reco_out_key', 'content_recommendations')
    user_ids = cfg['user_ids']
    user_ids = user_ids if isinstance(user_ids, list) else [user_ids]
    cold_start_user_id = 'cold_start_def_user'
    user_ids_es = user_ids + [cold_start_user_id]
    source = data.get("source", "email")
    top_recom = 50 if source == "email" else top_recom

    reco_results = {}
    if es.indices.exists(index=es_index):
        query = {
            'size': len(user_ids_es),
            'query':
                {
                    'terms': {
                        'reco_user_id.keyword': user_ids_es
                    }
                }
        }
        response = es.search(index=es_index, body=query)
        if len(response['hits']['hits']) > 0:
            try:
                for hit in response['hits']['hits']:
                    recos = hit['_source']['recommendations'][:top_recom]
                    user_id = hit['_source']['reco_user_id']
                    reco_results.update({user_id: recos})
            except:
                pass
    else:
        nlp_tags_logger.error(f"simpplr_get_reco Index: {es_index} doesn't exist")
        raise ValueError(f"HTTP Error 400: ES index not present, Missing {es_index}")

    if cold_start_user_id in reco_results:
        for user_id in user_ids:
            if user_id not in reco_results:
                reco_results.update({user_id: reco_results[cold_start_user_id]})
        reco_results.pop(cold_start_user_id)
    else:
        for user_id in user_ids:
            if user_id not in reco_results:
                reco_results.update({user_id: []})

    data[reco_out_key] = reco_results
    return data


def simpplr_content_reco_indexing(data, cfg):
    data = eapl_es_create_index(data, cfg)
    data = es_update(data, cfg)
    data = eapl_es_add_records(data, cfg)
    data = es_delete_record(data, cfg)
    return data


def simpplr_email_reco_filtering(data, cfg):
    input_key = cfg.get("input_key")
    out_key = cfg.get("out_key", input_key)
    email_reco_key = cfg.get("email_reco_key")
    org_id = cfg.get("org_id")
    source = data.get('source', 'email')
    top_recom = cfg.get('top_recom')
    reco_output = data[input_key]
    if source == 'email' and any(reco_output.values()):
        reco_count_fld = 'Simpplr__Content_Recommendation_Count__c'

        reco_df_out = pd.DataFrame(reco_output.items(), columns=['Simpplr__People__Id', 'recommendations'])
        reco_df = reco_df_out.explode('recommendations').dropna()
        reco_df = pd.concat([reco_df, reco_df['recommendations'].apply(pd.Series)], axis=1)
        mongodb_cfg = cfg['mongodb_connection']
        data = mongodb_connect(data, mongodb_cfg)
        reco_user_ids = cfg["query"]["Simpplr__People__Id"]["$in"]
        cfg["query"]["Simpplr__People__Id"]["$in"] = [reco_user_ids] if not isinstance(reco_user_ids,
                                                                                       list) else reco_user_ids
        data = mongodb_load_table(data, cfg)
        email_reco_df = data[email_reco_key]
        email_reco_df[reco_count_fld] = email_reco_df[reco_count_fld].astype(int)
        reco_df = reco_df.merge(email_reco_df, on=['Simpplr__People__Id', 'Simpplr__Content__Id'],
                                how="left")
        reco_df[reco_count_fld] = reco_df[reco_count_fld].fillna(0)
        reco_df = reco_df.query(f"{reco_count_fld} < 2 or reco_method == 'Must Read Reco'").groupby(
            'Simpplr__People__Id').head(top_recom)
        reco_df["Id"] = reco_df['Simpplr__Content__Id'] + '_' + reco_df['Simpplr__People__Id']
        current_datetime = dt.now()
        current_datetime = current_datetime.replace(tzinfo=timezone.utc)
        if reco_df.shape[0] > 0:
            upsert_list = list(reco_df.apply(lambda x: UpdateOne(
                {'Id': x['Id']},
                {
                    "$inc": {reco_count_fld: 1},
                    "$set": {"org_id": org_id,
                             "Simpplr__People__Id": x['Simpplr__People__Id'],
                             "Simpplr__Content__Id": x['Simpplr__Content__Id'],
                             "Simpplr__Recommendation_Feature__c": "email",
                             "updatedAt": current_datetime
                             },
                    "$setOnInsert": {"createdAt": current_datetime}

                },
                True), axis=1))
            data['write_input_data'] = upsert_list
            mongodb_write(data, cfg)

        reco_df = reco_df.groupby('Simpplr__People__Id')['recommendations'].apply(list).to_frame().reset_index()

        reco_df = reco_df_out[['Simpplr__People__Id']].merge(reco_df, on='Simpplr__People__Id', how='left')
        reco_df['recommendations'] = reco_df['recommendations'].fillna("").apply(list)
        reco_output = reco_df.set_index('Simpplr__People__Id').T.to_dict(orient='records')[0]

        data[out_key] = reco_output

    return data


simpplr_content_reco_fmap = {
    "simpplr_page_reco_data_prep": simpplr_page_reco_data_prep,
    "simpplr_data_harmonisation": simpplr_data_harmonisation,
    "simpplr_gen_reco": simpplr_gen_reco,
    "simpplr_get_reco": simpplr_get_reco,
    "simpplr_content_reco_indexing": simpplr_content_reco_indexing,
    "simpplr_email_reco_filtering": simpplr_email_reco_filtering
}
nlp_func_map.update(simpplr_content_reco_fmap)
