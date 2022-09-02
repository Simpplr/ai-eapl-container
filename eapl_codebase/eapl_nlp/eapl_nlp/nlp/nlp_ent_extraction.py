from pyate.term_extraction_pipeline import TermExtractionPipeline
from spacy.pipeline import EntityRuler
from nlp_rake import Rake
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import logging
import json
import copy
import pandas as pd

try:
    from .nlp_glob import *
    # from .nlp_ops import nlp_ops_funcs
    from .nlp_utils import *
except ImportError:
    from nlp_glob import *
    # from nlp_ops import nlp_ops_funcs
    from nlp_utils import *

nlp_ee_logger = logging.getLogger(__name__)


def er_lc_patterns(patterns):
    for each_pattern in patterns:
        tokens = each_pattern['pattern'].split(" ")
        p = [{"LOWER": tokens[0].lower()}]
        for i in range(1, len(tokens)):
            p = p + [{"IS_SPACE": True, 'OP': '?'}, {"IS_PUNCT": True, 'OP': '?'}, {"IS_SPACE": True, 'OP': '?'},
                     {"LOWER": tokens[i].lower()}]
        each_pattern['pattern'] = p
    return patterns


def er_lemma_patterns(nlp, patterns):
    for each_pattern in patterns:
        doc = nlp(each_pattern['pattern'].lower())
        p = [{"LEMMA": doc[0].lemma_}]
        for i in range(1, len(doc)):
            p = p + [{"IS_SPACE": True, 'OP': '?'}, {"IS_PUNCT": True, 'OP': '?'}, {"IS_SPACE": True, 'OP': '?'},
                     {"LEMMA": doc[i].lemma_}]
        each_pattern['pattern'] = p
    return patterns


def entityruler_init(data, cfg):
    nlp_key = cfg.get("nlp_key", "nlp")
    er_map_path_dct = cfg.get("er_map_path_dct")
    comp_name = cfg.get("comp_name", "eapl_ent_ruler")
    overwrite_ents = cfg.get("overwrite_ents", True)
    pattern_method = cfg.get("pattern_method", None)
    nlp = data[nlp_key]
    nlp_ee_logger.debug("Starting Entity Rule Init")
    existing_comps = [name for name, _ in nlp.pipeline]
    if comp_name not in existing_comps:
        nlp_ee_logger.debug(f"Adding {comp_name} to nlp pipeline")
        ruler = EntityRuler(nlp, overwrite_ents=overwrite_ents)
        with cwd(tempfile.mkdtemp()):
            eapl_download_file(data, er_map_path_dct)
            pat_fname = er_map_path_dct['dest_name']
            with open(pat_fname) as f:
                patterns = json.load(f)
        nlp_ee_logger.debug("Starting Add patterns")
        patterns_tmp = copy.deepcopy(patterns)
        if pattern_method == 'lowercase':
            patterns_tmp = er_lc_patterns(patterns_tmp)
        if pattern_method == 'lemma':
            patterns_tmp = er_lemma_patterns(nlp, patterns_tmp)
        if not overwrite_ents:
            patterns_tmp = patterns + patterns_tmp
        ruler.add_patterns(patterns_tmp)
        nlp.add_pipe(ruler, name=comp_name)
        nlp_glob[nlp_key] = nlp
    else:
        nlp_ee_logger.debug(f"{comp_name} already exists in nlp pipeline")
    nlp_ee_logger.debug("Ending Entity Rule Init")

    return data


def entityruler(data, text_dict, op_dict):
    doc_key = op_dict.get('doc_key', 'doc')
    out_key = op_dict.get('out_key', "er_tags")
    rem_ent_names = op_dict.get('rem_ent_names', [])
    doc = text_dict[doc_key]
    def_keep_ent_names = list(set(doc.ents))
    keep_ent_names = op_dict.get('keep_ent_names', def_keep_ent_names)

    er_tags = text_dict.get(out_key, [])

    for ent in doc.ents:
        try:
            if (ent.label_ in keep_ent_names) and (ent.label_ not in rem_ent_names):
                ed = {
                    'val': ent.text,
                    'mval': ent.ent_id_,
                    'label': ent.label_,
                    'method': 'entityruler'
                }
                if ed['mval'] == '':
                    ed['mval'] = ed['val']

                er_tags.append(ed)
        except Exception as e:
            print("Exception: ", e)
            pass

    text_dict[out_key] = er_tags
    return text_dict


def nounchunk_entities(data, text_dict, op_dict):
    doc_key = op_dict.get('doc_key', 'doc')
    out_key = op_dict.get('out_key', "nc_tags")
    doc = text_dict[doc_key]

    nc_tags = text_dict.get(out_key, [])
    for chunk in doc.noun_chunks:
        crd = chunk.root.dep_
        if crd in ['pobj', 'dobj']:
            nc_dct = {
                'val': chunk.text,
                'mval': chunk.text,
                'label': crd,
                'root': chunk.root.text,
                'method': 'noun_chunks_dp'
            }
            nc_tags.append(nc_dct)

    text_dict[out_key] = nc_tags
    return text_dict


def rake_init(data, cfg):
    rake_key = cfg.get('rake_key', 'rake')
    get_params = cfg.get('get_params', {})
    refresh = cfg.get('refresh', False)
    if refresh or rake_key not in nlp_glob.keys():
        rake = Rake(**get_params)
        nlp_glob[rake_key] = rake
    else:
        rake = nlp_glob[rake_key]
    data[rake_key] = rake
    return data


def rake_entities(data, text_dict, op_dict):
    text_key = op_dict.get('txt_key', 'txt')
    rake_key = op_dict.get('rake_key', 'rake')
    out_key = op_dict.get('out_key', "nc_tags")
    get_params = op_dict.get('get_params', {})
    filter_score = op_dict.get('filt_score', "score > 0")
    top_n = op_dict.get('top_n', 5)
    nc_tags = text_dict.get(out_key, [])
    text = text_dict[text_key]

    if rake_key in nlp_glob.keys():
        rake = nlp_glob[rake_key]
    else:
        rake = Rake(**get_params)
        nlp_glob[rake_key] = rake

    keywords = rake.apply((text))
    df = pd.DataFrame(keywords, columns=['tags', 'score'])
    df = df.query(filter_score)
    tags_lst = df['tags'].unique()
    tags_lst = tags_lst[:top_n]

    for tag in tags_lst:
        nc_dct = {
            'val': tag,
            'mval': tag,
            'label': "tag",
            'method': 'rake',
        }
        nc_tags.append(nc_dct)

    text_dict[out_key] = nc_tags

    return text_dict


def pyate_init(data, cfg):
    nlp_key = cfg.get("nlp_key", "nlp")
    comp_name = cfg.get("comp_name", "combo_basic")
    nlp = data[nlp_key]
    existing_comps = [name for name, _ in nlp.pipeline]

    if comp_name not in existing_comps:
        nlp.add_pipe(TermExtractionPipeline())
        nlp_glob[nlp_key] = nlp
        data[nlp_key] = nlp
        nlp_ee_logger.debug(f"Added {comp_name} to nlp pipeline")

    else:
        nlp_ee_logger.debug(f"{comp_name} already exists in nlp pipeline")

    return data


def pyate_entities(data, text_dict, op_dict):
    out_key = op_dict.get('out_key', "nc_tags")
    top_n = op_dict.get('top_n', 5)
    filter_score = op_dict.get('filt_score', "score > 4")
    doc_key = op_dict.get('doc_key', 'doc')
    nc_tags = text_dict.get(out_key, [])
    doc = text_dict[doc_key]
    series = doc._.combo_basic.sort_values(ascending=False)
    df = pd.DataFrame({'tags': series.index, 'score': series.values})
    df = df.query(filter_score)
    tags_lst = df['tags'].unique()
    tags_lst = tags_lst[:top_n]

    for tag in tags_lst:
        nc_dct = {
            'val': tag,
            'mval': tag,
            'label': "tag",
            'method': 'pyate',
        }
        nc_tags.append(nc_dct)

    text_dict[out_key] = nc_tags
    return text_dict


def keybert_entities(data, text_dict, op_dict):
    text_key = op_dict.get('txt_key', 'txt')
    keybert_key = op_dict.get('keybert_key', "keybert")
    out_key = op_dict.get('out_key', "nc_tags")
    get_params = op_dict.get('get_params', {'keyphrase_ngram_range': (1, 3), 'stop_words': None, 'top_n': 5})
    model_type = op_dict.get('model_type', 'sentence_model')
    model_name = op_dict.get('model_name', 'roberta-base-nli-stsb-mean-tokens')
    filter_score = op_dict.get('filt_score', "score > 0")
    nc_tags = text_dict.get(out_key, [])
    text = text_dict[text_key]

    if keybert_key in nlp_glob.keys():
        kw_model = nlp_glob[keybert_key]
    else:
        if model_type == 'sentence_model':
            model = SentenceTransformer(model_name)
        kw_model = KeyBERT(model=model)
        nlp_glob[keybert_key] = kw_model

    keywords = kw_model.extract_keywords(text, **get_params)

    df = pd.DataFrame(keywords, columns=['tags', 'score'])
    df = df.query(filter_score)
    tags_lst = df['tags'].unique()

    for tag in tags_lst:
        nc_dct = {
            'val': tag,
            'mval': tag,
            'label': "tag",
            'method': 'keybert',
        }
        nc_tags.append(nc_dct)

    text_dict[out_key] = nc_tags
    return text_dict


def eapl_combine_entities(data, text_dict, op_dict):
    ents_list_key = op_dict['ents_list_key']
    ents_def_method = op_dict.get('ents_def_method', 'default')
    mapped_ents_key = op_dict.get('mapped_ents_key', None)
    out_key = op_dict.get('out_key', mapped_ents_key)
    ents_list = text_dict[ents_list_key]
    mapped_ents = text_dict[mapped_ents_key].copy() if mapped_ents_key else []

    # doc = text_dict[doc_key]
    # er_tags = text_dict.get(out_key, [])

    val_list = [d['val'] for d in mapped_ents]
    for tag in ents_list:
        if tag not in val_list:
            tag_dct = {
                'val': tag,
                'mval': tag.lower(),
                'label': ents_def_method,
                'method': 'default'
            }
            mapped_ents.append(tag_dct)

    text_dict[out_key] = mapped_ents
    return text_dict


def filter_by_pos(data, text_dict, op_dict):
    nlp_key = op_dict.get("nlp_key", "nlp")
    text_key = op_dict.get('text_key', 'tags')
    rec_key = op_dict.get('rec_key', 'mval')
    doc_key = op_dict.get('doc_key', 'doc')
    rem_pos_tags = op_dict.get('rem_pos_tags', [])
    cleaned_tags = []

    all_tags = text_dict[text_key].copy()

    for tag in all_tags:
        cleaned_tag = True
        tag_val = tag[rec_key]
        doc = text_dict[doc_key]
        for token in doc:
            if token.text.lower() in tag_val.lower().split():
                if token.pos_ in rem_pos_tags:
                    cleaned_tag = False
        if tag not in cleaned_tags and cleaned_tag:
            cleaned_tags.append(tag)

    text_dict[text_key] = cleaned_tags

    return text_dict


def wit_entities_init(data, cfg):
    session_key = cfg.get("session_key", "wit_session")
    if session_key in nlp_glob.keys():
        wit_session = nlp_glob[session_key]
    else:
        wit_session = requests.Session()
        nlp_glob[session_key] = wit_session
    data[session_key] = wit_session
    return data


# Enhancement to be done by handling the version of wit as an input to the call.
def wit_entities(data, text_dict, op_dict):
    out_key = op_dict.get('out_key', "wit_tags")
    txt_key = op_dict.get('txt_key', "txt")
    session_key = op_dict.get('session_key', "wit_session")
    wit_auth_token = op_dict.get('wit_auth_token', 'VG3NAFQ567A3H4GKOVZHU4XDAB6OCCMC')
    w_tags = text_dict.get(out_key, [])
    raw_query = text_dict[txt_key]
    wit_session = data[session_key]
    headers = {
        'Authorization': 'Bearer ' + wit_auth_token,
    }
    params = (('q', raw_query), ('context', ""))
    try:
        response = wit_session.get('https://api.wit.ai/message', headers=headers, params=params)
        parsed = json.loads(response.content)
        entities = parsed['entities']

    except Exception as e:
        nlp_ee_logger.debug(f"No wit entities extracted {e}")
        entities = {}

    for x in entities:
        for y in range(len(entities[x])):
            wit_dict = {
                "val": entities[x][y]['body'],
                "mval": entities[x][y]['value'],
                "label": entities[x][y]['name'],
                "method": "wit"

            }
            if wit_dict['mval'] == '':
                wit_dict['mval'] = wit_dict['val']
            w_tags.append(wit_dict)
    text_dict[out_key] = w_tags
    return text_dict


nlp_ent_extraction_fmap = {
    'entityruler_init': entityruler_init,
    'entityruler': entityruler,
    'nounchunk_entities': nounchunk_entities,
    'eapl_combine_entities': eapl_combine_entities,
    'rake_entities': rake_entities,
    'pyate_entities': pyate_entities,
    'filter_by_pos': filter_by_pos,
    'wit_entities_init': wit_entities_init,
    'wit_entities': wit_entities,
    'keybert_entities': keybert_entities,
    'pyate_init': pyate_init,
    'rake_init': rake_init
}
nlp_func_map.update(nlp_ent_extraction_fmap)


def entityruler_config():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_pipe',
                       'entityruler_init',
                       'pyate_init',
                       'rake_init',
                       'wit_entities_init',
                       'record_nlp_ops',
                       'manage_data_keys'],

        'init_pipe': {
            'func': 'eapl_nlp_pipeline_init',
        },
        'pyate_init': {
            'func': 'pyate_init',
            'nlp_key': 'nlp',

        },
        'rake_init': {
            'func': 'rake_init',
            'get_params': {},
            'refresh': True
        },
        'entityruler_init': {
            'func': 'entityruler_init',
            "nlp_key": "nlp",
            "overwrite_ents": True,
            "pattern_method": 'lemma',
            'er_map_path_dct': {
                'file': 'https://s3.amazonaws.com/emplay.botresources/simpplr/simpplr_tags_map.json',
                'dest_name': 'psb_tag_map.json',
                'filemode': 'url'
            },
        },
        'wit_entities_init': {
            'func': 'wit_entities_init',
            'session_key': 'wit_session'
        },
        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    "op": "create_spacy_doc",
                },
                {
                    "op": "entityruler",
                    "doc_key": "doc",
                    "keep_ent_names": [
                        "PERSON",
                        "DATE",
                        "ORG",
                        "CARDINAL",
                        "TIME",
                        "WORK_OF_ART",
                        "MONEY",
                        "QUANTITY",
                        "ORDINAL",
                        "LAW",
                        "PERCENT"
                    ],
                    "rem_ent_names": [
                        "PERSON"
                    ],
                    "out_key": "tags"
                },
                {
                    "op": "nounchunk_entities",
                    'doc_key': 'doc',
                    'out_key': "tags"
                },
                {
                    "op": "wit_entities",
                    'txt_key': 'txt',
                    'out_key': 'tags'
                },
                {
                    "op": "rake_entities",
                    "txt_key": "txt",
                    'doc_key': 'doc',
                    'out_key': "tags",
                    "top_n": 5
                },
                {
                    "op": "pyate_entities",
                    "txt_key": "txt",
                    'doc_key': 'doc',
                    'out_key': "tags",
                    'top_n': 5
                },
                {
                    "op": "keybert_entities",
                    "txt_key": "txt",
                    'doc_key': 'doc',
                    'out_key': "tags"
                },
                # {
                #     "op": "eapl_combine_entities",
                #     'ents_list_key': 'tr_chunks',
                #     'ents_def_method': 'original',
                #     'mapped_ents_key': 'tags',
                #     'out_key': "m2_tags"
                # },
                {
                    "op": "filter_by_pos",
                    "text_key": "tags",
                    'doc_key': 'doc',
                    "rem_pos_tags": ["VERB", "PRON", "PRP$", "PRP", "JJR", "VBD", "RB", "DT", "VB", "VBZ", "VBG", "JJ",
                                     "conj"]
                },
                {
                    "op": "manage_text_dict_keys",
                    "pop_keys": ['doc']
                },
            ]
        },

        'manage_data_keys': {
            'func': 'manage_data_keys',
            'pop_keys': ['nlp']
        },
    }

    data = {
        'text_obj': [
            {
                "txt": """Customer Success - 2020 Q1 Update Q1 seems like a whirlwind for so many reasons. There was a meme on April 1st that said “Today is March 97th”. In some ways it felt like that as we embarked on an unprecedented situation. But, the reality was...it still only had 31 days and the quarter still only had 3 months. Yet...we still had a really productive quarter all the way around. We had significant accomplishments both with our renewals, customer go-lives, and initiatives that are helping us build repeatable and scalable processes. Highlights include: Our year-to-date renewal rate is 99.7% We had 25+ customer go-live 4 new members joined our team We implemented significant improvements to our implementation processes RENEWALS: The key renewals for Q1 were Pure Storage and Diligent plus we had 20 SMB accounts renew as well. Q1 has our most renewals (in terms of # of accounts) up for renewal this fiscal year. We lost one account overall (SMB) that unfortunately never went live due to a change in priorities and contacts. CUSTOMER GO-LIVES : We had over 25 go-lives across all of our segments Note: This is the most in any single quarter...ever! The Gold &amp; Platinum launches include: Pluralsight, Charter Manufacturing, Rivian, TripActions, Xilinx, Datto, Coursera, World Economic Forum and Topa (and also the relaunch of Workday!). TEAM GROWTH / STRUCTURE: We implemented our new team structure where we now have 3 distinct functions: Implementation / Customer Success / Support. To support this new structure: Implementation Team: Christine Robitaille transitioned to the Implementation Consultant role, while we brought in 2 new members for the other net-new roles on the team: Project Manager and Implementation Strategist ( Kristen van Eesteren and Christy Schoon respectively). Customer Success Team: We have 2 new CSMs for SMB and Enterprise ( Delil Martinez and Mike Messinger respectively). MOVE-THE-NEEDLE INITIATIVES Zendesk : Migrated from Desk.Com to Zendesk (read more details here ) Project Management Tool: We are managing all of our implementations with this new tool. This will really help us provide more structure and visibility into each project (both for us and our customers) and ensure we are properly scoping / charging for / staffing our projects. SOWs: Created 2 new comprehensive SOWs (one for SMB, one for Comm/Ent) Implementation Videos: Created 10 new videos used during implementation to help (a) scale the process (b) provide an improved experience for customers New “WFH” Service Package: Created, packaged (and delivered!) an accelerated services package for our COVID19 / WFH Relief Program. I am very proud of the execution across every team member. There was *a lot* going on all around us. Everyone really rose to the occasion as we all jumped in to hire, onboard, renew, implement, launch, train, build etc. And oh yeah...we did this all while bringing 2 new babies into the world! ( Mimi Kadash and Joseph Duffield ) A *HUGE* shout out to the new leadership on the CS Team ( Alan Daly and Kay Lim ). We could not have made this progress without you (and it is hard to believe this was your first quarter at Simpplr!) Onwards and Upwards for Q2. We have the momentum. We have the team. We are gettin’ the deals. Let’s go!""",

            }
        ]
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['text_obj'])

    return None


if __name__ == '__main__':
    entityruler_config()
