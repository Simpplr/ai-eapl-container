import pandas as pd
import regex as re
from collections import Counter
from bs4 import BeautifulSoup
import logging

from .nlp_glob import *
from .nlp_ops import *

nlp_tags_logger = logging.getLogger(__name__)


def simpplr_extract_tags(data, cfg):
    """Primary module for extracting tags from raw input text

    :param data: input data
    :type data: dict
    :param cfg: input config dict with keys:
                                        nlp_key: input spacy nlp object key
                                        input_key: list of dict objects key for input data processing
                                                   contains id, title, body/text_intro fields
                                        output_key: list of dict objects key for output of tagging algo
                                                   contains id, title, body/text_intro, article_body, list_of_tags fields
                                        rake_key: key for rake module object
                                        max_words: maximum words in key phrase

    :type cfg: dict
          contains configurable parameters to control the behaviour of algorithm
    :return: data
          contains list of dict items with input + tag related data structures
    :rtype: dict
    """
    input_key = cfg['input_key']
    text_ld = data[input_key]
    output_key = cfg.get('output_key', input_key)
    nlp_key = cfg['nlp_key']
    nlp = nlp_glob[nlp_key]
    rake_key = cfg['rake_key']
    rake_obj = nlp_glob[rake_key]
    max_words = cfg.get("max_words", 3)
    rem_ent_word = cfg.get("rem_ent_word", ". ")
    rep_pos_tags = [
        "VERB",
        "ADV",
        "PRON",
        "ADP"
    ]
    rep_ent_label_list = [
        "PERSON",
        "DATE",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "LANGUAGE",
        "NORP",
        "TIME",
        "GPE"
    ]

    def html_clean_text(html_content):
        """Clean the raw text data by removing html tags and patterns while retaining all the line breaks

        :param html_content: raw input text containing html patterns/tags
        :type html_content: str
        :return: text
        :rtype: str
        """
        elem = BeautifulSoup(html_content, features="html.parser")
        text = ''
        for e in elem.descendants:
            if isinstance(e, str):
                text += e
            elif e.name in ['br', 'p', 'h1', 'h2', 'h3', 'h4', 'tr', 'th']:
                text += '\n'
            elif e.name == 'li':
                text += '\n- '

        sents_nls = text.split('\n')
        sents = [f"{sent}." if re.search("[0-9a-zA-Z]+\s*$", sent) else sent for sent in sents_nls]
        text = "\n".join(sents)

        return text

    def replace_rem_ent_tags(txt):
        """Replaces the tokens with entities and pos tags present in rep_ent_label_list and rep_pos_tags with ". "

        :param txt: input text
        :type txt: str
        :return: txt
        :rtype: str
        """
        doc = nlp(txt)
        txt = ''
        mod_tok_list = []  # For detailed Analysis

        num_toks = len(doc)
        for ix, tok in enumerate(doc):
            mod_tok = tok.text_with_ws
            if tok.pos_ in rep_pos_tags or tok.ent_type_ in rep_ent_label_list or len(tok.text) > 15:
                mod_tok = rem_ent_word
            elif (ix <= num_toks - 3) and tok.pos_ == 'ADJ' and doc[ix + 1].pos_ in ['NOUN', 'PROPN', 'ADJ'] and \
                    doc[ix + 2].pos_ in ['NOUN', 'PROPN']:
                mod_tok = rem_ent_word
            # nlp_tags_logger.debug(f"pos: {tok.pos_} | tok: {tok.text_with_ws} | mod_tok: {mod_tok}")
            txt = txt + mod_tok
            mod_tok_list.append(mod_tok)  # For detailed Analysis

        return txt

    def get_word_count(txt):
        """counts the frequency of the words present in input text

        :param txt: input text
        :type txt: str
        :return: word_count_dict
        :rtype: dict
        """
        word_list = txt.lower().split()
        word_list = [w.strip(".,;:¡!¿?…⋯‹›«»\\'“”[]()⟨⟩}{&)") for w in word_list]
        word_count_dict = Counter(word_list)
        return word_count_dict

    def replace_date_time(txt):
        """replace all the date time patterns in text string with rem_ent_word

        :param txt: input text
        :type txt: str
        :return: txt
        :rtype: str
        """
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
            txt = re.sub(re.escape(t1), rem_ent_word, txt)

        return txt

    def extract_kws(txt):
        """Extracts keywords from input text with rake object

        :param txt: input text
        :type txt: str
        :return: kws
        :rtype: list
        """
        kws = rake_obj.apply(txt)
        return kws

    def rem_tags_with_rem(key_phrases, wc_dict):
        """Rerank the key phrases by normalizing the rake scores and blending with frequency of tokens

        :param key_phrases: input key phrases from rake
        :type key_phrases: list
        :param wc_dict: dictionary of the word count present in input text
        :type wc_dict: dict
        :return: tags
        :rtype: list
        """
        tags = []
        for kp in key_phrases:
            tag, score = kp
            num_toks = len(tag.split())
            if rem_ent_word not in tag and num_toks <= max_words:
                scale_factor = (num_toks ^ 2) / 4
                score = score / scale_factor if num_toks > 2 else score
                freq_score = sum([wc_dict[tok] for tok in tag.split()]) / num_toks
                # nlp_tags_logger.debug(f"{[(tok, wc_dict[tok]) for tok in tag.split()]}")
                # nlp_tags_logger.debug(f"nlp_rake score: {score} freq_score: {freq_score}")
                score = 0.9 * score + 0.1 * freq_score
                tags.append((tag, score))
        tags = sorted(tags, key=lambda x: x[1], reverse=True)
        return tags

    def tag_post_proc(kp_list):
        """Removes all the numerical phrases present in start and end of key phrase in key phrases list.
           Restrices tags length with max token length =< 15 and key phraseslength <40

        :param kp_list: list of key phrases
        :type kp_list: list
        :return: tags
        :rtype: list
        """
        tags = []
        for kp in kp_list:
            kp_toks = kp.split()
            if kp_toks:
                alnum_pat = '^[0-9a-zA-Z \-\']*$'
                kp_isalnum = re.match(alnum_pat, kp)
                kp_beg_end_num = max([kp_toks[0].isnumeric(), kp_toks[-1].isnumeric()])
                kp_beg_end_sc = max([bool(re.search('^[^a-zA-z0-9]|[^a-zA-z0-9]$', tok)) for tok in kp_toks])
                max_tok_len = max([len(tok) for tok in kp_toks])
                if bool(kp_isalnum) and (not kp_beg_end_num) and (not kp_beg_end_sc) and len(
                        kp) < 40 and max_tok_len <= 15:
                    tags.append(kp)
        return tags

    def map_match_tags(kp_list, rake_text):
        """Matches the list of tags with rake input text

        :param kp_list: list of keyphrases
        :type kp_list: list
        :param rake_text: input text of rake object
        :type rake_text: str
        :return: [{'val': kp, 'mval': match_case(kp, rake_text), 'label': 'tag', "method": "ent_rem_rake"}
                for kp in kp_list]
        :rtype: dict
        """

        def match_case(kp, text):
            try:
                kp = re.search(re.escape(kp), text, flags=re.IGNORECASE).group()
            except Exception as e:
                nlp_tags_logger.warning(f"Faced issues with matching pattern {kp}")
            return kp

        return [{'val': kp, 'mval': match_case(kp, rake_text), 'label': 'tag', "method": "ent_rem_rake"}
                for kp in kp_list]

    # Processing for all records for topic suggestion
    df = pd.DataFrame.from_records(text_ld)
    src_cols = list(df.columns)
    df['article_body'] = df.apply(lambda x: f"{x.title}. {x.text_intro}", axis=1)
    df['article_body'] = df['article_body'].apply(html_clean_text)
    nlp_tags_logger.debug(f"Topic Suggestion: Completed html to text")
    df['rake_text'] = df['article_body'].apply(replace_rem_ent_tags)
    nlp_tags_logger.debug(f"Topic Suggestion: Completed spacy doc and POS replacement")
    df['rake_text'] = df['rake_text'].apply(replace_date_time)
    df['wc_dict'] = df['article_body'].apply(get_word_count)
    nlp_tags_logger.debug(f"Topic Suggestion: Completed replace_date_time & word count")
    df['key_phrases'] = df['rake_text'].apply(extract_kws)
    nlp_tags_logger.debug(f"Topic Suggestion: Completed keyword extraction")
    df['non_rem_key_phrases'] = df.apply(lambda x: rem_tags_with_rem(x['key_phrases'], x['wc_dict']), axis=1)
    df['non_rem_kp_list_base'] = df['non_rem_key_phrases'].apply(
        lambda tag_score: [tag for tag, score in tag_score if score > 1])
    df['list_of_tags'] = df['non_rem_kp_list_base'].apply(tag_post_proc)
    df['tags'] = df.apply(lambda x: map_match_tags(x['list_of_tags'], x['rake_text']), axis=1)

    dst_cols = src_cols + ['article_body', 'list_of_tags', 'tags']
    df = df[dst_cols]
    data[output_key] = df.to_dict(orient='records')
    nlp_tags_logger.debug(f"Topic Suggestion: Completed post processing")
    return data


def simpplr_tags_post_proc(data, text_dict, op_dict):
    """filters the tags in list_of_tags to top_n

    :param text_dict: dictionary for input key
    :type text_dict: dict
    :param op_dict: config dictionary with keys:
                                           input key: contains input tags
                                           output key: contains output list of tags
                                           top_n: the top n tags numbers, defaults to 10
    :type op_dict: dict
    :return: text_dict
    :rtype: dict
    """
    top_n = op_dict.get("top_n", 10)
    input_key = op_dict['input_key']
    output_key = op_dict['output_key']
    tags_ld = text_dict[input_key]
    tags_l = []
    if len(tags_ld):
        tags_ld = tags_ld[:top_n]
        tags_l = [tag_dct['val'] for tag_dct in tags_ld]

    text_dict[input_key] = tags_ld
    text_dict[output_key] = tags_l
    return text_dict


simpplr_pproc_fmap = {
    "simpplr_extract_tags": simpplr_extract_tags,
    "simpplr_tags_post_proc": simpplr_tags_post_proc,
}
nlp_func_map.update(simpplr_pproc_fmap)
