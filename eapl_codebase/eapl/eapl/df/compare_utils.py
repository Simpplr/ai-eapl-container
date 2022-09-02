import pandas as pd
import numpy as np
import re


# Function to be used for exact match comparison between 2 dataframes. Even order of records will be considered.
def df_match_simple(df1, df2):
    return df1.equals(df2)


def df_match_onkeys(left_df, right_df, join, all_col, on_col):
    '''
    left_df : Dataframe for match used on left side of join on keys
    right_df : Dataframe for match used on right side of join on keys
    join : Type of join. Supports all acceptable values to 'how' field in pd.merge. Ex: 'inner', 'outer' etc.,
    all_col : List of columns for comparison including keys
    on_col : Keys on which to join.
    Requirement: The dataframe needs to be unique w.r.t to joint set of keys passed as argument with 'on_col' field
    Example Usage:
    test_dict = test_df_match(test_drr[list(port_links_drr.columns)], port_links_drr, 'left', list(port_links_drr.columns), ['SALES_REP_ID', 'OPP_ID', 'REGION'])
    '''
    on_col = [on_col] if type(on_col) is str else on_col

    # Prechecks for duplicate to avoid issues
    if (len(left_df[on_col].drop_duplicates()) != len(left_df[on_col])) \
            | (len(right_df[on_col].drop_duplicates()) != len(right_df[on_col])):
        print("There are primary key duplicates in the given dataframe.")
        return

    common_cols = list(set(left_df.columns).intersection(right_df.columns) - set(on_col))
    unmatched_col = []
    for col in common_cols:
        unmatched_col.append(col + '_left')
        unmatched_col.append(col + '_right')

    joined_df = pd.merge(left_df[all_col], right_df[all_col], how=join, indicator=True, on=on_col,
                         suffixes=['_left', '_right'])
    joined_df = joined_df.where(joined_df.notnull(), np.nan).sort_index(axis=1)
    joined_df = joined_df.replace('nan', np.nan)
    return_dict = dict()
    return_dict['main_dict'] = joined_df
    return_dict['on_key'] = on_col
    # unmatched_col = sorted(list(set(list(joined_df.columns))-set(all_col)))
    # unmatched_col.remove('_merge')
    unmatch = []
    match = []
    accuracy_total = []
    column = []
    nan_match_list = []
    non_nan_matches_list = []
    accuracy_notnan = []
    for col_index in range(0, len(unmatched_col), 2):
        df1 = joined_df[unmatched_col[col_index]]
        df2 = joined_df[unmatched_col[col_index + 1]]
        key_name = re.sub(r"\_left$", "", unmatched_col[col_index])
        if np.issubdtype(df1.dtype, np.number) & np.issubdtype(df2.dtype, np.number):
            unequal_df = joined_df[~(np.isclose(df1, df2, atol=1e-4, equal_nan=True))][
                on_col + [unmatched_col[col_index], unmatched_col[col_index + 1]]]
            unequal_df['_MATCH_TYPE'] = 'Not_Matching'
            equal_df = joined_df[(np.isclose(df1, df2, atol=1e-4, equal_nan=True))][
                on_col + [unmatched_col[col_index], unmatched_col[col_index + 1]]]
            equal_df['_MATCH_TYPE'] = 'Matching'
            nan_matches = sum(
                equal_df[unmatched_col[col_index]].isnull() & equal_df[unmatched_col[col_index + 1]].isnull())
        else:
            unequal_df = joined_df[((df1 != df2) & ~(df1.isnull() & df2.isnull()))][
                on_col + [unmatched_col[col_index], unmatched_col[col_index + 1]]]
            unequal_df['_MATCH_TYPE'] = 'Not_Matching'
            equal_df = joined_df[((df1 == df2) | ((df1 != df1) & (df2 != df2)))][
                on_col + [unmatched_col[col_index], unmatched_col[col_index + 1]]]
            equal_df['_MATCH_TYPE'] = 'Matching'
            nan_matches = sum(
                equal_df[unmatched_col[col_index]].isnull() & equal_df[unmatched_col[col_index + 1]].isnull())

        return_dict.update({key_name + '_matching': equal_df, key_name + '_not_matching': unequal_df})
        unmatch.append(len(unequal_df))
        match.append(len(equal_df))
        accuracy_total.append(float(len(equal_df)) / float(len(joined_df)) * 100)
        column.append(key_name)
        nan_match_list.append(nan_matches)
        non_nan_matches_list.append(len(equal_df) - nan_matches)
        accuracy_notnan.append(
            float(len(equal_df) - nan_matches) * 100 / float(len(equal_df) - nan_matches + len(unequal_df)) if float(
                len(equal_df) - nan_matches + len(unequal_df)) else 0)
    return_dict['comparison_matrix'] = pd.DataFrame(
        {'column': column, 'unmatch': unmatch, 'match': match, 'accuracy_total': accuracy_total,
         'nan_matches': nan_match_list, 'non_nan_matches': non_nan_matches_list, 'accuracy_notnan': accuracy_notnan})
    return_dict['key_list'] = list(return_dict.keys())
    return return_dict
