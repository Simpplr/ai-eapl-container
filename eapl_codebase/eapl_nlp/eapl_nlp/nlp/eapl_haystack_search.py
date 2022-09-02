import logging
import haystack
import time
from haystack import pipeline
import pandas as pd
import numpy as np

from haystack.file_converter.pdf import PDFToTextConverter
from haystack.file_converter.docx import DocxToTextConverter
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.preprocessor.preprocessor import PreProcessor
from haystack.schema import Document
from haystack.pipeline import JoinDocuments, TransformersQueryClassifier, SklearnQueryClassifier
from haystack.ranker.farm import FARMRanker

try:
    from .nlp_glob import nlp_func_map, nlp_glob
    from .nlp_ops import nlp_ops_funcs
    from .nlp_utils import nlp_utils_fmap
except ImportError:
    from nlp_glob import nlp_func_map, nlp_glob
    from nlp_ops import nlp_ops_funcs
    from nlp_utils import nlp_utils_fmap

eapl_hs_logger = logging.getLogger(__name__)

_docstore_func_map = {
    'ElasticsearchDocumentStore': haystack.document_store.elasticsearch.ElasticsearchDocumentStore,
    'FAISSDocumentStore': haystack.document_store.faiss.FAISSDocumentStore,
    'InMemoryDocumentStore': haystack.document_store.memory.InMemoryDocumentStore
}

_retriever_func_map = {
    'ElasticsearchRetriever': haystack.retriever.sparse.ElasticsearchRetriever,
    'DensePassageRetriever': haystack.retriever.dense.DensePassageRetriever,
    'EmbeddingRetriever': haystack.retriever.dense.EmbeddingRetriever,
    'TfidfRetriever': haystack.retriever.sparse.TfidfRetriever
}

_reader_func_map = {
    'FARMReader': haystack.reader.farm.FARMReader,
    'TransformersReader': haystack.reader.transformers.TransformersReader
}

_queryclassifier_func_map = {
    'SklearnQueryClassifier': haystack.pipeline.SklearnQueryClassifier,
    'TransformersQueryClassifier': haystack.pipeline.TransformersQueryClassifier
}

_callable_func_map = {
    # 'clean_wiki_text': haystack.preprocessor.cleaning.clean_wiki_text,
    'fetch_archive_from_http': haystack.preprocessor.utils.fetch_archive_from_http
}


def eapl_get_hs_obj_ds(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    hs_obj = data[hs_obj_key]
    docstore = hs_obj["docstore"]
    return hs_obj, docstore


def eapl_hs_init_setup(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    refresh = cfg.get('refresh', False)
    data[hs_obj_key] = nlp_glob.get(hs_obj_key, {})
    if hs_obj_key not in nlp_glob or refresh:
        hs_setup_pipeline = cfg.get("hs_setup_pipeline", [])
        for hs_cfg in hs_setup_pipeline:
            hs_cfg.update({'hs_obj_key': hs_obj_key})
            func_key = hs_cfg['func']
            eapl_hs_logger.debug(f"Processing {func_key} function")
            func = nlp_func_map[func_key]
            data = func(data, hs_cfg)
            nlp_glob[hs_obj_key] = data[hs_obj_key]
    else:
        data[hs_obj_key] = nlp_glob[hs_obj_key]

    return data


def eapl_hs_obj_mgmt(data, cfg):
    transfer = cfg.get('transfer', 'glob_to_data')
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')

    if transfer == 'glob_to_data':
        data[hs_obj_key] = nlp_glob.get(hs_obj_key, {})
    elif transfer == 'data_to_glob':
        nlp_glob[hs_obj_key] = data.get(hs_obj_key, {})

    return data


def eapl_hs_fetch_data(data, cfg):
    fetch_type = cfg.get('fetch_type', 'fetch_archive_from_http')
    fetch_params = cfg.get('fetch_params', {})

    fetch_func = _callable_func_map[fetch_type]
    fetch_func(**fetch_params)
    return data


def eapl_hs_preprocess(data, cfg):
    extract_type = cfg['extract_type']
    extract_params = cfg.get("extract_params", {})
    converter_params = cfg.get("converter_params", {})
    preprocessor_flag = cfg.get('preprocessor_flag', False)
    preprocessor_params = cfg.get('preprocessor_params', {})
    docs_key = cfg.get('docs_key', 'docs')

    docs = []
    if extract_type == 'DocxToTextConverter':
        converter = DocxToTextConverter(**extract_params)
        docs = converter.convert(**converter_params)
    elif extract_type == 'PDFToTextConverter':
        converter = PDFToTextConverter(**extract_params)
        docs = converter.convert(**converter_params)
    elif extract_type == 'convert_files_to_dicts':
        conv_params = converter_params.copy()
        if 'clean_func' in conv_params:
            clean_func_key = conv_params['clean_func']
            conv_params['clean_func'] = _callable_func_map[clean_func_key]
        docs = convert_files_to_dicts(**conv_params)
    else:
        eapl_hs_logger.debug(f"Currently {extract_type} is not supported")
    eapl_hs_logger.debug(f"Extracted {len(docs)} from the inputs provided")

    if preprocessor_flag:
        processor = PreProcessor(**preprocessor_params)
        docs = processor.process(docs)
        eapl_hs_logger.debug(f"After document preprcoessing: {len(docs)} available for indexing")

    data[docs_key] = docs
    return data


def eapl_hs_docstore(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    hs_obj = data[hs_obj_key]
    docstore_filepath = cfg.get('docstore_filepath', None)
    docstore_type = cfg['docstore_type']
    docstore_init_type = cfg.get('docstore_init_type', 'init')  # Options: 'init', 'load'
    docstore_params = cfg.get('docstore_params', {}).copy()
    docstore_load_params = cfg.get('docstore_load_params', None)
    delete_all_docs_flag = cfg.get('delete_all_docs_flag', False)
    delete_params = cfg.get('delete_params', {})
    data_docstore_params_key = cfg.get('data_docstore_params_key', None)

    if data_docstore_params_key:
        data_docstore_params = data[data_docstore_params_key]
        docstore_params.update(data_docstore_params)

    docstore_func = _docstore_func_map[docstore_type]
    if docstore_init_type == 'init':
        docstore = docstore_func(**docstore_params)

    if docstore_init_type == 'load' and docstore_load_params and docstore_type in ['FAISSDocumentStore']:
        docstore = docstore_func.load(**docstore_load_params)

    if delete_all_docs_flag:
        eapl_hs_logger.debug(f"Started deleting all records in doc_store in func: eapl_hs_docstore")
        docstore.delete_documents(**delete_params)

    hs_obj["docstore_filepath"] = docstore_filepath
    hs_obj["docstore"] = docstore
    return data


def eapl_hs_copy_hsparams(data, cfg):
    hs_keys = cfg.get('hs_keys', ["docstore"])
    ref_hs_obj_key = cfg['ref_hs_obj_key']
    ref_hs_obj = data[ref_hs_obj_key]
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    hs_obj = data[hs_obj_key]

    for key in hs_keys:
        hs_obj[key] = ref_hs_obj[key]
    return data


def eapl_hs_delete_documents(data, cfg):
    hs_obj, docstore = eapl_get_hs_obj_ds(data, cfg)
    delete_params = cfg.get('delete_params', {})
    docstore.delete_documents(**delete_params)
    return data


def eapl_hs_df2docs(data, cfg):
    df_key = cfg['df_key']
    df = data[df_key]
    docs_key = cfg.get('docs_key', 'docs')
    rename_cols = cfg.get('rename_cols', None)
    text_fields = cfg.get('text_fields', ['text'])
    meta_data_fields = cfg.get('meta_data_fields', None)
    add_meta_field_params = cfg.get('add_meta_field_params', None)
    join_str = cfg.get('join_str', '. ')

    if df.shape[0] > 0:
        df_index = df.copy()
        if rename_cols:
            df_index = df_index.rename(columns=rename_cols)

        df_index['text'] = df[text_fields].apply(lambda row: join_str.join(row.values.astype(str)), axis=1)
        if add_meta_field_params:
            for key, value in add_meta_field_params:
                df_index[key] = value

        if meta_data_fields:
            doc_fields = ['text'] + meta_data_fields
            df_index = df_index[doc_fields]

        df_index.fillna('', inplace=True)
        recs = df_index.to_dict('records')
        docs = [Document.from_dict(rec) for rec in recs]

    else:
        docs = []

    data[docs_key] = docs
    return data


def eapl_hs_docs2recs(data, cfg):
    docs_key = cfg.get('docs_key', 'docs')
    docs = data[docs_key]
    recs_key = cfg.get('recs_key', 'recs')
    keep_keys = cfg.get('keep_keys', ['id', 'text', 'score', 'probability', 'question'])
    float_keys = cfg.get('float_keys', ['score'])
    recs = []
    for rec_inp in docs:
        if isinstance(rec_inp, Document):
            rec_inp = rec_inp.to_dict()
        rec = {k: v for k, v in rec_inp.items() if k in keep_keys}
        rec.update(rec_inp['meta'])
        for key in float_keys:
            rec[key] = float(rec[key])
        recs.append(rec)

    data[recs_key] = recs
    return data


def eapl_hs_write_documents(data, cfg):
    hs_obj, docstore = eapl_get_hs_obj_ds(data, cfg)
    retriever = hs_obj.get("retriever", None)
    retriever_type = hs_obj.get("retriever_type", None)
    update_embedding = cfg.get('update_embedding', True)
    update_embeddings_params = cfg.get('update_embeddings_params', {})
    write_documents_params = cfg.get('write_documents_params', {})
    delay = cfg.get('delay', 0)
    docs_key = cfg.get('docs_key', "docs")

    dicts = data[docs_key]
    docstore.write_documents(dicts, **write_documents_params)
    time.sleep(delay)
    if retriever_type in ['DensePassageRetriever', 'EmbeddingRetriever'] and update_embedding is True:
        eapl_hs_logger.debug(f"Update of embeddings started in eapl_hs_retriever")
        docstore.update_embeddings(retriever=retriever, **update_embeddings_params)
        eapl_hs_logger.debug(f"Update of embeddings completed in eapl_hs_retriever")

    if hs_obj["docstore_filepath"]:
        docstore.save(hs_obj["docstore_filepath"])
    return data


def eapl_hs_retriever(data, cfg):
    hs_obj, docstore = eapl_get_hs_obj_ds(data, cfg)
    retriever_type = cfg['retriever_type']
    retriever_params = cfg.get('retriever_params', {})

    retriever_func = _retriever_func_map[retriever_type]
    retriever = retriever_func(document_store=docstore, **retriever_params)

    hs_obj["retriever_type"] = retriever_type
    hs_obj["retriever"] = retriever
    return data


def eapl_hs_ranker(data, cfg):
    hs_obj, docstore = eapl_get_hs_obj_ds(data, cfg)
    ranker_params = cfg.get('ranker_params', {})

    ranker = FARMRanker(**ranker_params)
    hs_obj["ranker"] = ranker
    return data


def eapl_hs_get_all_docs(data, cfg):
    hs_obj, docstore = eapl_get_hs_obj_ds(data, cfg)
    out_key = cfg.get('out_key', "docs_gad")
    get_docs_params = cfg.get('get_docs_params', {})
    method = cfg.get('method', "get_all_documents")

    if method == "get_document_by_id":
        docs_gad = [docstore.get_document_by_id(**get_docs_params)]
    elif method == "get_documents_by_id":
        docs_gad = docstore.get_documents_by_id(**get_docs_params)
    else:
        docs_gad = docstore.get_all_documents(**get_docs_params)
    data[out_key] = docs_gad
    return data


def eapl_hs_get_sim_docs(data, cfg):
    hs_obj, docstore = eapl_get_hs_obj_ds(data, cfg)
    out_key = cfg.get('out_key', "sim_docs")
    query_by_embed_params = cfg.get('query_by_embed_params', {})
    doc_data_key = cfg.get('doc_data_key', "docs_gad")
    get_docs_params = cfg.get('get_docs_params', {})
    docs_gad = docstore.get_all_documents(**get_docs_params) if get_docs_params else data[doc_data_key]

    result_list = []
    for d in docs_gad:
        try:
            query_emb = d.embedding
        except:
            raise ValueError(f"HTTP Error 400: Embedding for given id not found")

        query_by_embed_params.update({"query_emb": query_emb})
        results = docstore.query_by_embedding(**query_by_embed_params)
        doc_list = []

        # temp code to handle non serializable object
        for res in results:
            res = res.__dict__
            res['score'] = float(res['score'])  # raise an issue
            doc_list.append(res)

        result_list.append(doc_list)
    data[out_key] = result_list

    return data


def eapl_hs_reader(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    hs_obj = data[hs_obj_key]
    reader_type = cfg.get('reader_type', 'FARMReader')
    reader_params = cfg.get('reader_params', {})

    reader_func = _reader_func_map[reader_type]
    reader = reader_func(**reader_params)
    hs_obj["reader"] = reader
    return data


def eapl_hs_queryclassifier(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    hs_obj = data[hs_obj_key]
    qclassifier_type = cfg.get('qclassifier_type', 'SklearnQueryClassifier')
    qclassifier_params = cfg.get('qclassifier_params', {})

    qclassifier_func = _queryclassifier_func_map[qclassifier_type]
    qclassifier = qclassifier_func(**qclassifier_params)
    hs_obj["qclassifier"] = qclassifier
    return data


def eapl_hs_joindocuments(data, cfg):
    hs_obj_key = cfg['hs_obj_key']
    hs_obj = data[hs_obj_key]
    join_params = cfg.get('join_params', {})

    join_obj = JoinDocuments(**join_params)
    hs_obj["join_obj"] = join_obj
    return data


def eapl_hs_custompipeline(data, cfg):
    hs_pipe_cfg = cfg.get("hs_setup_pipeline", [])
    pipe = pipeline.Pipeline()
    for hs_cfg in hs_pipe_cfg:
        hs_obj_key = hs_cfg['hs_obj_key']
        component = hs_cfg["component"]
        hs_obj = data[hs_obj_key]
        compent_obj = hs_obj[component]
        name = hs_cfg["name"]
        inputs = hs_cfg["inputs"]
        eapl_hs_logger.debug(f"hs_obj_key: {hs_obj_key} | component: {component} | name: {name} | inputs: {inputs}")
        pipe.add_node(component=compent_obj, name=name, inputs=inputs)
    return pipe


def eapl_hs_create_pipe(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    hs_obj = data[hs_obj_key]
    pipe_type = cfg['pipe_type']
    # To be supported : GenerativeQAPipeline, SearchSummarizationPipeline
    if pipe_type == 'DocumentSearchPipeline':
        pipe = pipeline.DocumentSearchPipeline(retriever=hs_obj["retriever"])
    elif pipe_type == 'FAQPipeline':
        pipe = pipeline.FAQPipeline(retriever=hs_obj["retriever"])
    elif pipe_type == 'ExtractiveQAPipeline':
        pipe = pipeline.ExtractiveQAPipeline(reader=hs_obj["reader"], retriever=hs_obj["retriever"])
    elif pipe_type == 'CustomPipeline':
        pipe = eapl_hs_custompipeline(data, cfg)
    else:
        eapl_hs_logger.debug(f"Currently {pipe_type} is not supported")

    hs_obj["pipe"] = pipe
    return data


def eapl_hs_rerank_sr(data, cfg):
    sr_key = cfg.get('sr_key', 'search_results')
    output_key = cfg.get('output_key', sr_key)
    n = cfg.get('n', 5)
    rank_var = cfg.get("rank_var", 'score')
    doc_id_var = cfg.get("doc_id_var", "org_id")
    method_var = cfg.get("method_var", "method")
    score_filter = cfg.get("min_score", "score > 0")
    wt = cfg.get("weights", [1, 1, 1, 1])
    wt_clm = cfg.get("weights_cols",
                     ["sap_io_body_embed", "sap_io_st_embed", "sap_io_th_embed", "sap_io_title_embed"])
    wt = [w / sum(wt) for w in wt]
    search_res = data[sr_key]
    df = pd.DataFrame.from_records(search_res)
    match_cols = cfg.get("match_cols", [doc_id_var, 'question', 'Content_Title', 'Content_Text', 'Title_Hierarchy'])
    match_df = df[match_cols].drop_duplicates(subset=doc_id_var)

    df_sel = df
    groupby = cfg.get("groupby", [doc_id_var, method_var])
    df_sel = df_sel.sort_values(by=groupby + [rank_var], ascending=[True, True, False])
    df_sel = df_sel.drop_duplicates(subset=groupby)

    df_sent_cols = [doc_id_var, 'text']
    df_sent = df_sel.query(f"{method_var} == 'sap_io_st_embed'")[df_sent_cols]
    df_sent.rename(columns={'text': 'sent'}, inplace=True)

    df_pivot = pd.pivot_table(df_sel, values=[rank_var], index=[doc_id_var], columns=[method_var],
                              aggfunc=np.sum).reset_index()
    ren_col_names = df_pivot.columns
    ren_cols = [val[1] if val[1] != '' else val[0] for val in ren_col_names]
    df_pivot.columns = ren_cols

    df_pivot_min = df_pivot.transform(lambda x: x.fillna(x.min()))

    df_pivot_min["max_score"] = df_pivot_min[wt_clm].max(axis=1)

    eval_str = f"0.5 * ({' + '.join([str(wt[i]) + ' * ' + str(wt_clm[i]) for i in range(len(wt))])} + max_score )"
    df_pivot_min["score"] = df_pivot_min.eval(eval_str)
    df_pivot_min = df_pivot_min.query(score_filter)

    df_search_final = df_pivot_min.sort_values(by=['score'], ascending=[False]).head(n)

    df_search_final = df_search_final.merge(match_df, on=[doc_id_var], how='left')
    df_search_final = df_search_final.merge(df_sent, on=[doc_id_var], how='left')
    data[output_key] = df_search_final.to_dict(orient='records')

    return data


def eapl_hs_pipe_run(data, cfg):
    hs_obj_key = cfg.get('hs_obj_key', 'hs_obj')
    hs_obj = data[hs_obj_key]
    pipe_params = cfg.get('pipe_params', {})
    query_key = cfg.get('query_key', 'query')
    query = data[query_key]
    results_key = cfg.get('results_key', 'hs_results')

    pipe = hs_obj["pipe"]
    res = pipe.run(query=query, **pipe_params)
    try:
        if 'documents' in res:
            data[results_key] = res['documents']
        elif 'answers' in res:
            data[results_key] = res['answers']
    except:
        eapl_hs_logger.debug(f"Exception during piperun. No results found")
        data[results_key] = []

    return data


eapl_haystack_search_fmap = {
    'eapl_hs_init_setup': eapl_hs_init_setup,
    'eapl_hs_obj_mgmt': eapl_hs_obj_mgmt,
    'eapl_hs_fetch_data': eapl_hs_fetch_data,
    'eapl_hs_preprocess': eapl_hs_preprocess,
    'eapl_hs_docstore': eapl_hs_docstore,
    'eapl_hs_copy_hsparams': eapl_hs_copy_hsparams,
    'eapl_hs_delete_documents': eapl_hs_delete_documents,
    'eapl_hs_df2docs': eapl_hs_df2docs,
    'eapl_hs_docs2recs': eapl_hs_docs2recs,
    'eapl_hs_write_documents': eapl_hs_write_documents,
    'eapl_hs_retriever': eapl_hs_retriever,
    'eapl_hs_ranker': eapl_hs_ranker,
    'eapl_hs_get_all_docs': eapl_hs_get_all_docs,
    'eapl_hs_get_sim_docs': eapl_hs_get_sim_docs,
    'eapl_hs_reader': eapl_hs_reader,
    'eapl_hs_create_pipe': eapl_hs_create_pipe,
    'eapl_hs_pipe_run': eapl_hs_pipe_run,
    'eapl_hs_joindocuments': eapl_hs_joindocuments,
    'eapl_hs_rerank_sr': eapl_hs_rerank_sr,
    "eapl_hs_queryclassifier": eapl_hs_queryclassifier
}
nlp_func_map.update(eapl_haystack_search_fmap)


def test_hs():
    return None


if __name__ == '__main__':
    test_hs()
