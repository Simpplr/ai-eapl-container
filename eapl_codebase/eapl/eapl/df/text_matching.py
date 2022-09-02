import pandas as pd
from fuzzywuzzy import fuzz, process


def fuzzy_text_match(txt_frm_match, txt_to_match, match_func=fuzz.ratio, score=85, limit=5):
    '''
    txt_frm_match : List of text strings for matching
    txt_to_match  : List of text strings to find similarity with txt_frm_match
    txt_frm_match and txt_to_match should be utf-8 compliance
    match_func : Fuzzy match algo function name. Supported functions are : 
                 fuzz.ratio,fuzz.partial_ratio,token_sort_ratio,partial_token_sort_ratio
                 for details refer to https://github.com/seatgeek/fuzzywuzzy/blob/master/fuzzywuzzy/fuzz.py
    score = Minimum match score. Takes value between 0 & 100
    limit = Max number of best matches meeting score threshold. Limit should be > 0
    '''
    fuzz_match = {tfm: process.extractBests(tfm, txt_to_match, score_cutoff=score, scorer=match_func, limit=limit) for
                  tfm in txt_frm_match}
    # Eliminate dictionary items with zero values
    fuzz_match = {k: v for k, v in list(fuzz_match.items()) if v != []}
    match_df = pd.DataFrame()
    match_df = pd.concat(
        [match_df.append([[k, m, s]], ignore_index=False) for k, v in list(fuzz_match.items()) for m, s in v])
    match_df.reset_index(drop=True, inplace=True)
    match_df.columns = ['txt_frm_match', 'txt_to_match', 'match_score']
    return match_df
