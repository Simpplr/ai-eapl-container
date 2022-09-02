import pandas as pd
import pytz
import numpy as np
import regex as re
from dateutil import parser
from datetime import datetime as dt
from sklearn import preprocessing
import logging

from .nlp_glob import *
from .nlp_ops import *

nlp_tags_logger = logging.getLogger(__name__)


def clean_text(data, text_dict, op_dict):
    input_key = op_dict.get('input_key', 'txt')
    output_key = op_dict.get('output_key', 'txt')
    txt = text_dict[input_key]
    ops_list = op_dict.get('ops_list')

    for proc_op_dict in ops_list:
        op = proc_op_dict['op']

        if op == 'remove_dates':
            try:
                nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero
                nMNTH = r'(?:11|12|10|0?[1-9])'  # month can be 1 to 12 with a leading zero
                nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis
                # that people doon't generally use all number format for old dates, but write them out
                nDELIM = r'(?:[\/\-\._])?'  #
                NUM_DATE = f"""
                    (?P<num_date>
                        (?:^|\D) # new bit here
                        (?:
                        # YYYY-MM-DD
                        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
                        |
                        # YYYY-DD-MM
                        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
                        |
                        # DD-MM-YYYY
                        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
                        |
                        # MM-DD-YYYY
                        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
                        )
                        (?:\D|$) # new bit here
                    )"""
                DAY = r"""
                (?:
                    # search 1st 2nd 3rd etc, or first second third
                    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth)
                    |
                    # or just a number, but without a leading zero
                    (?:[123]?\d)
                )"""
                MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'
                YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""
                DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'

                YEAR_4D = r"""(?:[12]\d\d\d)"""
                DATE_PATTERN = f"""(?P<wordy_date>
                    # non word character or start of string
                    (?:^|\W)
                        (?:
                            # match various combinations of year month and day
                            (?:
                                # 4 digit year
                                (?:{YEAR_4D}{DELIM})?
                                    (?:
                                    # Day - Month
                                    (?:{DAY}{DELIM}{MONTH})
                                    |
                                    # Month - Day
                                    (?:{MONTH}{DELIM}{DAY})
                                    )
                                # 2 or 4 digit year
                                (?:{DELIM}{YEAR})?
                            )
                            |
                            # Month - Year (2 or 3 digit)
                            (?:{MONTH}{DELIM}{YEAR})
                        )
                    # non-word character or end of string
                    (?:$|\W)
                )"""

                TIME = r"""(?:
                (?:
                # first number should be 0 - 59 with optional leading zero.
                [012345]?\d
                # second number is the same following a colon
                :[012345]\d
                )
                # next we add our optional seconds number in the same format
                (?::[012345]\d)?
                # and finally add optional am or pm possibly with . and spaces
                (?:\s*(?:a|p)\.?m\.?)?
                )"""

                COMBINED = f"""(?P<combined>
                    (?:
                        # time followed by date, or date followed by time
                        {TIME}?{DATE_PATTERN}{TIME}?
                        |
                        # or as above but with the numeric version of the date
                        {TIME}?{NUM_DATE}{TIME}?
                    )
                    # or a time on its own
                    |
                    (?:{TIME})
                )"""

                myResults = []
                myDate = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)
                for matchGroup in myDate.finditer(txt):
                    myResults.append(matchGroup.group('combined'))

                for t1 in myResults:
                    txt = re.sub(t1, " ", txt)
            except:
                nlp_tags_logger.info("failed regex case")
                pass

        elif op == 'remove_url':
            url_pattern = "((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?Ã‚Â«Ã‚Â»Ã¢â‚¬Å“Ã¢â‚¬ÂÃ¢â‚¬ËœÃ¢â‚¬â„¢]))"
            txt = re.sub(url_pattern, "", txt)

    text_dict[output_key] = txt
    return text_dict


def merge_results(data, cfg):
    lista = cfg.get('lista', [])
    listb = cfg.get('listb', [])
    output_key = cfg.get("output_key", "results")
    unq_ids_key = cfg.get("unq_ids_key", "unq_ids_lst")
    id_col = cfg.get("id_col", "id")

    lst = []

    a = data[lista]
    b = data[listb]

    for i in range(0, len(a)):
        lst = lst + [a[i] + b[i]]

    for i in range(len(lst)):
        l2 = []
        for j in lst[i]:
            if j not in l2:
                l2.append(j)
        lst[i] = l2

    flat_list = [item for sublist in lst for item in sublist]
    values_of_key = [a_dict[id_col] for a_dict in flat_list]
    data[output_key] = lst
    data[unq_ids_key] = values_of_key

    return data


def eapl_reranking(data, cfg):
    input_key = cfg.get("input_key", "similar_text")
    df = pd.DataFrame(data[input_key])
    ops_list = cfg.get('ops_list', None)
    if not df.empty:
        for proc_op_dict in ops_list:
            op = proc_op_dict['op']
            if op == "normalize_values":
                col_list = proc_op_dict['col_list']
                for col in col_list:
                    method = col['method']
                    col_name = col['field_name']
                    output_key = col.get("output_field_name", col_name + "_score")
                    if method == "max_norm":
                        df['temp'] = df[col_name].apply(pd.to_numeric)
                        df[output_key] = df['temp'].apply(
                            lambda x: (x - min(df['temp'])) / (max(df['temp']) - min(df['temp'])))
                    if method == "skl_norm":
                        df['temp'] = df[col_name].apply(pd.to_numeric)
                        df[output_key] = pd.Series(preprocessing.normalize([df['temp']])[0])
                    if method == "recency_norm":
                        def_date = "1970-01-01T00:00:00.000+0000"
                        if col_name not in df.columns:
                            df[col_name] = def_date
                        df[col_name] = np.where(df[col_name] != " ", df[col_name], def_date)
                        df[output_key] = df[col_name].apply(lambda x: (dt.now(tz=pytz.UTC) - parser.parse(str(x))).days)
                        df[output_key] = df[output_key].rank(ascending=False, na_option='bottom')
                        df[output_key] = pd.Series(preprocessing.normalize([df[output_key]])[0])
                data[input_key] = df.to_dict(orient='records')
            if op == "weighted_average":
                sort_desc = proc_op_dict.get("sort_desc", True)
                output_key = proc_op_dict.get("output_key", "weighted_score")
                out_key_val = proc_op_dict.get("out_key_val", input_key)
                input_key_val = proc_op_dict["input_key_val"]
                threshold = proc_op_dict.get("threshold", 0)
                df[output_key] = df[input_key_val.keys()].apply(
                    lambda x: sum([x[i] * input_key_val[i] for i in input_key_val.keys()]), axis=1)
                df[output_key] = pd.Series(preprocessing.normalize([df[output_key]])[0])
                df = df.sort_values(by=output_key, ascending=not sort_desc).query('score > @threshold')
                data[out_key_val] = df.to_dict(orient='records')
            if op == "top_n":
                sort_by_col = proc_op_dict.get("sort_by_col", "weighted_average")
                top_n = proc_op_dict.get("top_n", 10)
                sort_desc = proc_op_dict.get("sort_desc", True)
                df = df.sort_values(sort_by_col, ascending=not sort_desc).head(top_n)
                data[out_key_val] = df.to_dict(orient='records')
    else:
        nlp_tags_logger.debug("eapl_reranking: No Recommendations found for this user Id, Returning empty list")

    return data


def simpplr_basic_criteria_checks(data, cfg):
    """Check for different basic criteria for the data and keys that are passed in the cfg

    :param data: Relevant objects for validation specified in the config parameter
    :type data: dict

    :param cfg: Contains the following keys which are checked for validity
        - :primary_keys dict: Checks if the mandatory keys are present in the required tables
        - :value_sync_list list: Checks if the given keys have the smae value or not
        - :text_key str: key from data object for basic checks
    :type cfg: dict

    :return data: Checks for differenct cases and if nothing fails data is passed as is, else an error is raised based on the conditions
    """
    primarykeys = cfg.get("primarykeys", {}).copy()  # mandatory keys with datatype
    exc_emptykeys = cfg.get("exc_emptykeys", [])
    text_key = cfg.get("text_key", "text_obj")
    substitutions_key = cfg.get("substitutions_key", "substitutions")
    value_check_dict = cfg.get("value_check_dict", {})
    value_sync_list = cfg.get("value_sync_list", [])
    datatype_dict = {"str": str, "int": int, "dict": dict, "list": list, "set": set, "bool": bool}
    primarykeys.update({key: datatype_dict[val] for key, val in primarykeys.items()})
    input_data = data[text_key]
    if text_key == substitutions_key:
        input_data = [input_data]
    for input_record in input_data:
        if set(primarykeys.keys()).issubset(input_record.keys()):
            for key in primarykeys.keys():
                if not input_record[key] and key not in exc_emptykeys:
                    raise ValueError(f"HTTP Error 400: {key} null/nan/empty value")
                else:
                    input_record[key] = primarykeys[key](input_record[key])
        else:
            missing_keys = list(set(primarykeys.keys()) - set(input_record.keys()))
            raise KeyError(f"HTTP Error 400: Mandatory keys not present {missing_keys[0]}")

        if value_sync_list:
            for lt in value_sync_list:
                if not (input_record[lt[0]].lower() == input_record[lt[1]].lower()):
                    raise ValueError(f"HTTP Error 400: {lt[0], lt[1]} not in sync")

        if value_check_dict:
            for ckey, cval in value_check_dict.items():
                if not cval[0] < int(input_record[ckey]) <= cval[1]:
                    raise ValueError(f"HTTP Error 400: {ckey} value not in range")

    if text_key == substitutions_key:
        input_data = input_data[0]
    data[text_key] = input_data
    return data


simpplr_utils_fmap = {
    "clean_text": clean_text,
    "merge_results": merge_results,
    "eapl_reranking": eapl_reranking,
    "simpplr_basic_criteria_checks": simpplr_basic_criteria_checks,
}
nlp_func_map.update(simpplr_utils_fmap)
