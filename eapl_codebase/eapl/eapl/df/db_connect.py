import pandas as pd
import logging

db_con_logger = logging.getLogger(__name__)


def get_data_frm_db_or_pkl(connection=None, query=None, fetch_db=False, pkl_file=None, colcase='UPPER'):
    db_con_logger.debug("Started fetching data")
    if fetch_db:
        df = pd.read_sql(query, connection)
    else:
        df = pd.read_pickle(pkl_file) if pkl_file else None

    if pkl_file and fetch_db: df.to_pickle(pkl_file)

    if colcase is None or colcase.upper() == 'UPPER':
        df.columns = df.columns.str.upper()
    else:
        df.columns = df.columns.str.lower()
    if not df.empty: return df
    db_con_logger.debug("Ended fetching data")
    return df


# Function to get max timestamp among managed tables with RSOURCE_ID column
# Ex: get_max_tbl_upd_time(connection, ['sales_crmdatamaster', 'sales_quotadatamaster'])
def get_max_tbl_upd_time(connection, tables):
    query = '''
            SELECT MAX(DMERGED) DMERGED
            FROM SALES_DATAFILES
            WHERE ID IN (SELECT DISTINCT RSOURCE_ID FROM {table})
                  AND DMERGED IS NOT NULL
            '''
    max_ts_df = None
    for tbl in tables:
        q = query.format(table=tbl)
        ts_df = get_data_frm_db_or_pkl(connection, query=q, fetch_db=True)
        ts_df['DMERGED'] = pd.to_datetime(ts_df['DMERGED'], errors='coerce')
        if max_ts_df is None:
            max_ts_df = ts_df
        elif ts_df['DMERGED'][0] > max_ts_df['DMERGED'][0]:
            max_ts_df = ts_df

    return max_ts_df


# Function to get max timestamp among Risk Reco tables with TIMESTAMP column
# Ex: get_max_upd_time_reco_tbl(connection, ['REGION'], 'TIMESTAMP_VAR', 'sales_dealriskreco')
def get_max_upd_time_reco_tbl(connection, grp_by_cols, ts_col, tbl, upd_ts_col):
    grp_by = None
    if grp_by_cols is None:
        query = 'SELECT MAX({ts_col}) {upd_ts_col} FROM {tbl}'
    else:
        grp_by = ",".join(grp_by_cols)
        query = '''SELECT {grp_by}, MAX({ts_col}) {upd_ts_col} 
               FROM {tbl}
               GROUP BY {grp_by}'''

    q = query.format(grp_by=grp_by, ts_col=ts_col, tbl=tbl, upd_ts_col=upd_ts_col)
    ts_df = get_data_frm_db_or_pkl(connection, query=q, fetch_db=True)
    ts_df[upd_ts_col] = pd.to_datetime(ts_df[upd_ts_col], errors='coerce')
    return ts_df
