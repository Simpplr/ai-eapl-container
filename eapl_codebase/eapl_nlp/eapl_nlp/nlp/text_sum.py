from operator import itemgetter
from math import sqrt
import pytextrank
import logging

try:
    from .nlp_glob import *
    from .nlp_ops import nlp_ops_funcs
except ImportError:
    from nlp_glob import *
    from nlp_ops import nlp_ops_funcs

nlp_p_logger = logging.getLogger(__name__)


def eapl_txt_sum_init(data, cfg):
    nlp_key = cfg.get('nlp_key', 'nlp')
    comp_name = cfg.get("comp_name", "textrank")
    nlp = data[nlp_key]
    existing_comps = [name for name, _ in nlp.pipeline]
    if comp_name not in existing_comps:
        nlp_p_logger.debug("Adding {comp_name} to nlp pipeline")
        tr = pytextrank.TextRank()
        nlp.add_pipe(tr.PipelineComponent, name=comp_name, last=True)
        nlp_glob[nlp_key] = nlp
    else:
        nlp_p_logger.debug("{comp_name} already exists in nlp pipeline")

    data[nlp_key] = nlp
    return data


def eapl_textrank_txt_summary(doc, limit_phrases=4, limit_sentences=2):
    tr_chunks = []
    sent_bounds = [[s.start, s.end, set([])] for s in doc.sents]

    phrase_id = 0
    unit_vector = []

    for p in doc._.phrases:
        # nlp_p_logger.debug(f"{phrase_id} | {p.text} | {p.rank}")
        tr_chunks.append(p.text)
        unit_vector.append(p.rank)
        for chunk in p.chunks:
            # nlp_p_logger.debug(f"Chunk range: {chunk.start} {chunk.end}")
            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.start <= sent_end:
                    # nlp_p_logger.debug(f" {sent_start} | {chunk.start} | {chunk.end} | {sent_end}")
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if phrase_id == limit_phrases:
            break

    sum_ranks = sum(unit_vector)
    unit_vector = [rank / sum_ranks for rank in unit_vector]

    sent_rank = {}
    sent_id = 0

    for sent_start, sent_end, sent_vector in sent_bounds:
        # nlp_p_logger.debug(sent_vector)
        sum_sq = 0.0

        for phrase_id in range(len(unit_vector)):
            # nlp_p_logger.debug(f"{phrase_id} {unit_vector[phrase_id]}")

            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id] ** 2.0

        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1

    sorted(sent_rank.items(), key=itemgetter(1))

    sent_text = {}
    sent_id = 0
    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    num_sent = 0
    sum_txt_lst = []
    for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
        # nlp_p_logger.debug(f"{sent_id} {sent_text[sent_id]}")
        num_sent += 1
        sum_txt_lst.append(sent_text[sent_id])

        if num_sent == limit_sentences:
            break

    sum_txt = " ".join(sum_txt_lst)
    # nlp_p_logger.debug(f"Summarized Text : {sum_txt}")
    return sum_txt, tr_chunks


def text_summarization(data, text_dict, op_dict):
    doc_key = op_dict.get('doc_key', 'doc')
    txt_sum_key = op_dict.get('txt_sum_key', 'txt_summary')
    tr_chunks_key = op_dict.get('tr_chunks_key', 'tr_chunks')
    limit_phrases = op_dict.get('limit_phrases', 4)
    limit_sentences = op_dict.get('limit_sentences', 2)

    doc = text_dict[doc_key]
    txt_summary, tr_chunks = eapl_textrank_txt_summary(doc, limit_phrases, limit_sentences)

    text_dict[txt_sum_key] = txt_summary
    text_dict[tr_chunks_key] = tr_chunks
    return text_dict


text_sum_fmap = {
    'eapl_txt_sum_init': eapl_txt_sum_init,
    'text_summarization': text_summarization
}
nlp_func_map.update(text_sum_fmap)


def test_txt_sum_config():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['init_pipe', 'eapl_txt_sum_init', 'record_nlp_ops', 'manage_data_keys'],

        'init_pipe': {
            'func': 'eapl_nlp_pipeline_init',
        },

        'eapl_txt_sum_init': {
            'func': 'eapl_txt_sum_init',
        },

        'record_nlp_ops': {
            'func': 'eapl_nlp_record_process',
            'ops': [
                {
                    "op": "create_spacy_doc",
                },
                {
                    "op": "text_summarization",
                    'doc_key': 'doc',
                    'txt_sum_key': 'txt_summary',
                    'tr_chunks_key': 'tr_chunks',
                    'limit_phrases': 5,
                    'limit_sentences': 3,
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
                "txt": """Sushant Singh Rajput passed away on June 14 at his Bandra residence. Since then, his fans and family members were demanding for a CBI inquiry. Recently, the Supreme Court announced its verdict and transferred the case to the CBI. The team of Central Bureau of Investigation is currently in Mumbai and they are actively investigating the actor’s death case. The CBI team recently questioned Sushant’s flatmate Siddharth Pithani and his cook Neeraj since they were present at the residence on June 14. Now according to the latest report by Times Now, Rhea Chakraborty’s brother Showik Chakraborty is currently being grilled by the CBI at the DRDO guesthouse."""
            },
            {
                "txt": """A single-day spike of 61,408 infections has taken India's coronavirus caseload to 31,06,348, data from the Union Health Ministry this morning showed. The number of deaths climbed to 57,542 with 836 people dying of the disease in a span of 24 hours. Recoveries among COVID-19 patients in the country surged to 23,38,035, pushing India's recovery rate to 75.27 per cent. Maharashtra accounts for the maximum of cases (6,82,383), followed by Tamil Nadu (3,79,385), Andhra Pradesh (3,53,111), Karnataka (2,77,814), Uttar Pradesh (1,87,781) and Delhi (1,61,466). India is the country with the third-highest coronavirus caseload in the world after the United States and Brazil."""
            },
        ]
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['text_obj'])

    return None


if __name__ == '__main__':
    test_txt_sum_config()
