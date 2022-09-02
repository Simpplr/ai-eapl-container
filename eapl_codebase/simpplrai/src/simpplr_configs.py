import os
import ast
import logging
import environ
from datetime import datetime, timedelta
import pytz

simpplrai_code_version = "1.0.19"
nlp_simpplr_cfg_logger = logging.getLogger(__name__)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# env_path = f"{BASE_DIR}/.env"
# if not os.path.exists(env_path):
#     env_path = ".env"
# nlp_simpplr_cfg_logger.info(f".env file path: {env_path}")

env = environ.Env()
# env.read_env(env_path)

def_redis_init_params = {}  
def_mdb_endpoint_params = {} 
def_es_endpoint_params = {} 

redis_init_params = ast.literal_eval(env.get_value('REDIS_INIT_PARAMS', default=str(def_redis_init_params)))
mdb_endpoint_params = ast.literal_eval(env.get_value('DB_MONGO_ENDPOINT_PARAMS', default=str(def_mdb_endpoint_params)))
es_endpoint_params = ast.literal_eval(env.get_value('ES_ENDPOINT_PARAMS', default=str(def_es_endpoint_params)))

current_time = datetime.now(tz=pytz.UTC)
days_to_subtract = 365
train_from_timestamp = current_time - timedelta(days=days_to_subtract)

# Common Configs for reuse across multiple post requests
simpplr_generic_cfgs = {
    "init_nlp_md": {
        "func": "eapl_nlp_pipeline_init",
        "model": "en_core_web_md",
        "nlp_key": "nlp_md"
    },
    "start_redis": {
        "func": "start_redis",
        "redis_init_params": redis_init_params
    },
    "hs_init_es_search": {
        "func": "eapl_hs_init_setup",
        "hs_obj_key": "hs_rc_test",
        "hs_setup_pipeline": [
            {
                "func": "eapl_hs_docstore",
                "docstore_type": "ElasticsearchDocumentStore",
                "docstore_params": {
                    "duplicate_documents": "overwrite",
                    "custom_mapping": {
                        "mappings": {
                            "properties": {
                                "expires_at": {
                                    "type": "date",
                                    "ignore_malformed": True
                                },
                                "text_embed": {
                                    "type": "text"
                                },
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": 768
                                }
                            }
                        }
                    },
                    "return_embedding": True,
                    "similarity": "cosine",
                    "index": "rc_test",
                    "excluded_meta_data": [
                        "text_embed",
                        "text"
                    ],
                    **es_endpoint_params
                },
                "delete_all_docs_flag": False
            },
            {
                "func": "eapl_hs_retriever",
                "retriever_type": "EmbeddingRetriever",
                "retriever_params": {
                    "embedding_model": "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                    "model_format": "sentence_transformers",
                    "use_gpu": False
                }
            }
        ]
    },
    "hs_init_es_generic": {
        "func": "eapl_hs_init_setup",
        "hs_obj_key": "hs_pr_init_es",
        "hs_setup_pipeline": [
            {
                "func": "eapl_hs_docstore",
                "docstore_type": "ElasticsearchDocumentStore",
                "docstore_params": {
                    "index": "pr_init_es",
                    **es_endpoint_params
                }
            }
        ]
    },
    "init_es": {
        "func": "init_es",
        "es_params": es_endpoint_params,
        "es_obj": "es_obj"
    },
    "mongodb_connection": {
        "func": "mongodb_connect",
        "endpoint": mdb_endpoint_params,
        "database_name": "${mongodb_database_name}",
        "refresh": "${mongodb_refresh}",
        "output_key": "mongodb"
    },
    "init_use_model_st": {
        "func": "eapl_use_init_sentence_transformers",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_key": "use_model_sentencetransformers"
    }
}
nlp_simpplr_cfg_logger.debug(f"{mdb_endpoint_params} set as mongodb instance")
nlp_simpplr_cfg_logger.debug(f"{es_endpoint_params} set as elasticsearch instance")
nlp_simpplr_cfg_logger.info(f"{redis_init_params} set as redis instance")

simpplr_config = {
    "topic_suggestion": {
        "config_seq": [
            "import_funcs",
            "input_data_check",
            "init_use_model_st",
            "init_pipe",
            "rake_init",
            "simpplr_extract_tags",
            "semantic_drop_duplicate",
            "manage_data_keys"
        ],
        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                "simpplr_extract_tags": "from .simpplr_tags_script import simpplr_pproc_fmap",
                "eapl_semantic_drop_duplicate": "from .text_embed import use_embed_map",
                "entityruler_init": "from .nlp_ent_extraction import nlp_ent_extraction_fmap"
            }
        },
        "input_data_check": {
            "func": "simpplr_basic_criteria_checks",
            "primarykeys": {"title": "str", "text_intro": "str"},
            "exc_emptykeys": ["text_intro"],
            "text_key": "text_obj"
        },
        "init_use_model_st": simpplr_generic_cfgs["init_use_model_st"],
        "init_pipe": simpplr_generic_cfgs["init_nlp_md"],
        "rake_init": {
            "func": "rake_init",
            "rake_key": "rake",
            "get_params": {
                "min_chars": 4,
                "max_words": 5
            }
        },
        "simpplr_extract_tags": {
            "func": "simpplr_extract_tags",
            "nlp_key": "nlp_md",
            "input_key": "text_obj",
            "output_key": "text_obj",
            "rake_key": "rake",
            "max_words": 5
        },
        "semantic_drop_duplicate": {
            "func": "eapl_nlp_record_process",
            "text_obj": "text_obj",
            "ops": [
                {
                    "op": "eapl_semantic_drop_duplicate_st",
                    "text_data_type": "list_of_dicts",
                    "model_key": "use_model_sentencetransformers",
                    "text_key": "mval",
                    "input_key": "tags",
                    "diverse": 0.75,
                    "out_key": "tags",
                    "unq_val_key": "val"
                },
                {
                    "op": "simpplr_tags_post_proc",
                    "input_key": "tags",
                    "output_key": "list_of_tags",
                    "top_n": 10
                }
            ]
        },
        "manage_data_keys": {
            "func": "manage_data_keys",
            "keep_keys": ['text_obj']
        }
    },
    "post_data_redis": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "data_substitutions_check",
                "input_data_check",
                "start_redis",
                "save_data_redis",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "start_redis": "from .simpplr_redis_ops import simpplr_redis_ops_fmap"
                }
            },
            "data_substitutions_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"org_id": "str", "index": "str"},
                "value_sync_list": [['org_id', 'index']],
                "text_key": "substitutions"
            },
            "input_data_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"id": "str", "title": "str", "text_intro": "str", "type": "str",
                                "site_type": "str"},
                "exc_emptykeys": ["text_intro"],
                "text_key": "text_obj"
            },
            "start_redis": simpplr_generic_cfgs["start_redis"],
            "save_data_redis": {
                "func": "save_data_redis",
                "input_key": "text_obj",
                "org_id": "rc_${index}",
                "id_col": "id"
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "pop_keys": [
                    "text_obj"
                ]
            }
        }
    },
    "recommendations_setup": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "data_substitutions_check",
                "init_pipe",
                "hs_init_es_search",
                "start_redis",
                "read_data_redis",
                "replace_nans",
                "record_nlp_ops",
                "eapl_data_fmt_conv2",
                "hs_create_index_data",
                "write_documents",
                "redis_get_ids",
                "flush_redis_data",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap",
                    "eapl_data_fmt_conv": "from .eapl_kpi_ref import eapl_kpi_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "start_redis": "from .simpplr_redis_ops import simpplr_redis_ops_fmap"
                }
            },
            "data_substitutions_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"org_id": "str", "index": "str"},
                "value_sync_list": [['org_id', 'index']],
                "text_key": "substitutions"
            },
            "init_pipe": {
                "func": "eapl_nlp_pipeline_init",
                "model": "en_core_web_sm",
                "nlp_key": "nlp"
            },
            "hs_init_es_search": simpplr_generic_cfgs["hs_init_es_search"],
            "start_redis": simpplr_generic_cfgs["start_redis"],
            "read_data_redis": {
                "func": "read_data_redis",
                "input_key": "text_obj",
                "org_id": "rc_${index}",
            },
            "replace_nans": {
                "func": "eapl_kpi_non_df_ops",
                "cfg_exec_cond": "len(data['text_obj']) > 0",
                "df_func": "eapl_derived_cols_wrapper",
                "input_key": "text_obj",
                "output_key": "text_obj",
                "ops": [
                    ["df_getattr", ["eval", {"expr": "doc_id = id"}]],
                    ["df_getattr", ["fillna", {"value": " "}]],
                ]
            },
            "record_nlp_ops": {
                "func": "eapl_nlp_record_process",
                "text_obj": "text_obj",
                "ops": [
                    {
                        "op": "eapl_str_ops",
                        "input_key": "text_intro",
                        "output_key": "text_intro",
                        "ops_list": [
                            {
                                "op": "re_replace",
                                "match_pat": "<.*?>|&nbsp;|&#39|&gt|&lt",
                                "rep_pattern": " "
                            }
                        ]
                    },
                    {
                        "op": "eapl_str_ops",
                        "input_key": "text_intro",
                        "output_key": "cleaned_text_intro",
                        "ops_list": [
                            {
                                "op": "re_replace",
                                "match_pat": "[12][0-9]{3}",
                                "rep_pattern": " "
                            },
                            {
                                "op": "eapl_record_eval_ops",
                                "eval_expr": "str(cleaned_text_intro)[:500]"
                            },
                            {
                                "op": "strip"
                            },
                            {
                                "op": "re_replace",
                                "match_pat": "\\s\\s+",
                                "rep_pattern": " "
                            }
                        ]
                    },
                    {
                        "op": "clean_text",
                        "op_exec_cond": "cleaned_text_intro is not None",
                        "input_key": "cleaned_text_intro",
                        "output_key": "cleaned_text_intro",
                        "ops_list": [
                            {
                                "op": "remove_dates"
                            }
                        ]
                    },
                    {
                        "op": "eapl_str_ops",
                        "input_key": "cleaned_text_intro",
                        "output_key": "cleaned_text_intro",
                        "ops_list": [
                            {
                                "op": "strip"
                            },
                            {
                                "op": "re_replace",
                                "match_pat": "\\s\\s+",
                                "rep_pattern": " "
                            }
                        ]
                    },
                    {
                        "op": "eapl_str_ops",
                        "input_key": "title",
                        "output_key": "cleaned_title",
                        "ops_list": [
                            {
                                "op": "re_replace",
                                "match_pat": "<.*?>|&nbsp;|&#39|&gt|&lt|[12][0-9]{3}|\\s\\s+",
                                "rep_pattern": " "
                            },
                            {
                                "op": "strip"
                            }
                        ]
                    },
                    {
                        "op": "clean_text",
                        "op_exec_cond": "cleaned_title is not None",
                        "input_key": "cleaned_title",
                        "output_key": "cleaned_title",
                        "ops_list": [
                            {
                                "op": "remove_dates"
                            }
                        ]
                    },
                    {
                        "op": "eapl_str_ops",
                        "input_key": "cleaned_title",
                        "output_key": "cleaned_title",
                        "ops_list": [
                            {
                                "op": "strip"
                            },
                            {
                                "op": "re_replace",
                                "match_pat": "\\s\\s+",
                                "rep_pattern": " "
                            }
                        ]
                    },
                    {
                        "op": "eapl_record_eval_ops",
                        "eval_expr": "str(cleaned_title) +'   '+str(cleaned_text_intro)+'   '+str(type)",
                        "output_key": "text_embed"
                    },
                    {
                        "op": "manage_text_dict_keys",
                        "pop_keys": [
                            "cleaned_text_intro",
                            "cleaned_title"
                        ]
                    }
                ]
            },
            "eapl_data_fmt_conv2": {
                "func": "eapl_data_fmt_conv",
                "input_key": "text_obj",
                "format_conv": "records_to_df",
                "output_key": "ref_df"
            },
            "hs_create_index_data": {
                "func": "eapl_hs_df2docs",
                "df_key": "ref_df",
                "docs_key": "docs",
                "text_fields": [
                    "text_embed"
                ],
                "meta_data_fields": None
            },
            "write_documents": {
                "func": "eapl_hs_write_documents",
                "hs_obj_key": "hs_rc_test",
                "docs_key": "docs",
                "write_documents_params": {"index": "rc_${index}"},
                "update_embeddings_params": {"update_existing_embeddings": False, "index": "rc_${index}"}
            },
            "redis_get_ids": {
                "func": "redis_get_ids",
                "org_id": "rc_${index}",
                "id_col": "id",
                "out_key": "redis_ids"
            },
            "flush_redis_data": {
                "func": "rc_flush_redis_ids",
                "data_ids_key": "redis_ids",
                "hs_obj_key": "hs_rc_test",
                "get_docs_params": {
                    "index": "rc_${index}"
                },
                "method": "get_documents_by_id",
                "out_key": "docs_gad",
                "meta_key_name": "doc_id",
                "org_id": "rc_${index}",
                "id_col": "id",
                "id_val_key": "doc_id_val"
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": [
                    "data"
                ]
            }
        }
    },
    "real_time_reco": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "data_substitutions_check",
                "hs_init_es_search",
                "eapl_hs_get_all_docs",
                "eapl_hs_get_sim_docs1",
                "eapl_hs_get_sim_docs2",
                "combine_results",
                "eapl_hs_docs2recs",
                "reco_to_df",
                "rename_col",
                "convert_df_to_recs",
                "rerank_records",
                "format_data",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap",
                    "eapl_data_fmt_conv": "from .eapl_kpi_ref import eapl_kpi_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "start_redis": "from .simpplr_redis_ops import simpplr_redis_ops_fmap"
                }
            },
            "data_substitutions_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"org_id": "str", "index": "str", "id": "str"},
                "value_sync_list": [['org_id', 'index']],
                "text_key": "substitutions"
            },
            "hs_init_es_search": simpplr_generic_cfgs["hs_init_es_search"],
            "eapl_hs_get_all_docs": {
                "func": "eapl_hs_get_all_docs",
                "method": "get_document_by_id",
                "hs_obj_key": "hs_rc_test",
                "get_docs_params": {
                    "id": "${id}",
                    "index": "rc_${index}",
                },
                "out_key": "doc_data"
            },
            "eapl_hs_get_sim_docs1": {
                "func": "eapl_hs_get_sim_docs",
                "hs_obj_key": "hs_rc_test",
                "query_by_embed_params": {
                    "return_embedding": False,
                    "top_k": 5,
                    "index": "rc_${index}",
                    "filters": {
                        "site_type": [
                            "private",
                            "unlisted"
                        ]
                    }
                },
                "out_key": "similar_text1",
                "id_col": "doc_id",
                "doc_data_key": "doc_data"
            },
            "eapl_hs_get_sim_docs2": {
                "func": "eapl_hs_get_sim_docs",
                "hs_obj_key": "hs_rc_test",
                "query_by_embed_params": {
                    "return_embedding": False,
                    "top_k": 5,
                    "index": "rc_${index}",
                    "filters": {
                        "site_type": [
                            "public"
                        ]
                    }
                },
                "out_key": "similar_text2",
                "id_col": "doc_id",
                "doc_data_key": "doc_data"
            },
            "combine_results": {
                "func": "eapl_eval_ops",
                "eval_expr": "[(similar_text1[x]+similar_text2[x]) for x in range(0,len(similar_text1))][0]",
                "output_key": "similar_text"
            },
            "eapl_hs_docs2recs": {
                "func": "eapl_hs_docs2recs",
                "docs_key": "similar_text",
                "recs_key": "similar_text",
                "keep_keys": [
                    "text",
                    "probability",
                    "question",
                    "score",
                    "id"
                ],
                "float_keys": [
                    "score"
                ]
            },
            "reco_to_df": {
                "func": "eapl_data_fmt_conv",
                "input_key": "similar_text",
                "format_conv": "records_to_df",
                "output_key": "similar_text_df"
            },
            "rename_col": {
                "eval_expr": "similar_text_df.eval('similarity_score = score')",
                "func": "eapl_eval_ops",
                "output_key": "similar_text_df"
            },
            "convert_df_to_recs": {
                "func": "eapl_data_fmt_conv",
                "input_key": "similar_text_df",
                "output_key": "similar_text",
                "format_conv": "df_to_records"
            },
            "rerank_records": {
                "func": "eapl_reranking",
                "input_key": "similar_text",
                "ops_list": [
                    {
                        "op": "normalize_values",
                        "col_list": [
                            {
                                "field_name": "publishStartDate",
                                "method": "recency_norm",
                                "output_field_name": "recency_rate"
                            }
                        ]
                    },
                    {
                        "op": "weighted_average",
                        "input_key_val": {
                            "similarity_score": 0.3,
                            "recency_rate": 0.15
                        },
                        "sort_desc": True,
                        "output_key": "score"

                    }
                ]
            },
            "format_data": {
                "func": "eapl_eval_ops",
                "eval_expr": "[similar_text]",
                "output_key": "similar_text"
            },
            "manage_dict_keys": {
                "func": "eapl_nlp_record_process",
                "text_obj": "similar_text",
                "ops": [
                    {
                        "op": "manage_text_dict_keys",
                        "pop_keys": [
                            "embedding"
                        ]
                    }
                ]
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": [
                    "similar_text"
                ]
            }
        }
    },
    "remove_unpublished_content": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "data_substitutions_check",
                "start_redis",
                "hs_init_es_generic",
                "init_es",
                "es_delete_record",
                "refresh_index",
                "search_es",
                "delete_data_redis",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "start_redis": "from .simpplr_redis_ops import simpplr_redis_ops_fmap",
                    "init_es": "from .elastic_search_custom import es_search_fmap"
                }
            },
            "data_substitutions_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"org_id": "str", "index": "str", "id": "str"},
                "value_sync_list": [['org_id', 'index']],
                "text_key": "substitutions"
            },
            "start_redis": simpplr_generic_cfgs["start_redis"],
            "hs_init_es_generic": simpplr_generic_cfgs["hs_init_es_generic"],
            "init_es": simpplr_generic_cfgs["init_es"],
            "es_delete_record": {
                "func": "es_delete_record",
                "index": "rc_${index}",
                "es_obj": "es_obj",
                "del_body": {
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "term": {
                                        "_id": "${id}"
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "refresh_index": {
                "func": "refresh_index",
                "index": "rc_${index}",
                "es_obj": "es_obj"
            },
            "search_es": {
                "func": "search_es",
                "index": "rc_${index}",
                "es_obj": "es_obj",
                "size": 20,
                "search_body": {
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "term": {
                                        "_id": "${id}"
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "delete_data_redis": {
                "func": "delete_data_redis",
                "id_col": "id",
                "id_val": "${id}",
                "org_id": "rc_${index}",
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": [
                    "es_index_delete",
                    "es_index_update",
                    "es_res",
                    "search_es",
                    "data"
                ]
            }
        }
    },
    "remove_expired_content": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "data_substitutions_check",
                "hs_init_es_generic",
                "init_es",
                "es_update",
                "es_delete_record",
                "refresh_index",
                "search_es",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "start_redis": "from .simpplr_redis_ops import simpplr_redis_ops_fmap",
                    "init_es": "from .elastic_search_custom import es_search_fmap"
                }
            },
            "data_substitutions_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"org_id": "str", "index": "str"},
                "value_sync_list": [['org_id', 'index']],
                "text_key": "substitutions"
            },
            "hs_init_es_generic": simpplr_generic_cfgs["hs_init_es_generic"],
            "init_es": simpplr_generic_cfgs["init_es"],
            "refresh_index": {
                "func": "refresh_index",
                "index": "rc_${index}",
                "es_obj": "es_obj"
            },
            "es_update": {
                "func": "es_update",
                "index": "rc_${index}",
                "es_obj": "es_obj",
                "update_body": {
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "range": {
                                        "expires_at": {
                                            "lt": "now"
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "script": {
                        "expiry_status": "expired"
                    }
                }
            },
            "es_delete_record": {
                "func": "es_delete_record",
                "index": "rc_${index}",
                "es_obj": "es_obj",
                "del_body": {
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "range": {
                                        "expires_at": {
                                            "lt": "now"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "search_es": {
                "func": "search_es",
                "index": "rc_${index}",
                "es_obj": "es_obj",
                "size": 20,
                "search_body": {
                    "query": {
                        "bool": {
                            "filter": [
                                {
                                    "range": {
                                        "expires_at": {
                                            "lt": "now"
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": [
                    "es_index_delete",
                    "es_index_update",
                    "es_res"
                ]
            }
        }
    },
    "page_reco_setup": {
        "job_type": "background",
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "data_substitutions_check",
                "init_es",
                "mongodb_connection",
                "mongodb_load_tables",
                "data_harmonisation",
                "user_data_prep",
                "data_prep",
                "init_model",
                "train_model",
                "gen_reco",
                "index_reco_records",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "init_es": "from .elastic_search_custom import es_search_fmap",
                    "simpplr_email_reco_filtering": "from .simpplr_content_reco import simpplr_content_reco_fmap",
                    "mongodb_connect": "from .simpplr_mongodb_connection import eapl_simpplr_mogodb_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "eapl_implicit_data_prep": "from .reco_implicit import collab_implicit_fmap",
                }
            },
            "data_substitutions_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"org_id": "str", "index": "str", "mongodb_database_name": "str",
                                "mongodb_refresh": "bool"},
                "exc_emptykeys": ["mongodb_refresh"],
                "value_sync_list": [['org_id', 'index']],
                "text_key": "substitutions"
            },
            "init_es": simpplr_generic_cfgs["init_es"],
            "mongodb_connection": simpplr_generic_cfgs["mongodb_connection"],
            "mongodb_load_tables": {
                "func": "mongodb_load_mul_table",
                "mongo_db_key": "mongodb",
                "tables_info": [
                    {
                        "collection_name": "Simpplr__Simpplr__Content__c",
                        "output_key": "content_raw_df",
                        "query": {
                            "org_id": "${org_id}",
                            "Simpplr__Is_Published__c": "true"
                        },
                        "projection": {
                            "Simpplr__Content__Id": 1,
                            "Simpplr__Is_Published__c": 1,
                            "Simpplr__Event_End_DateTime__c": 1,
                            "Simpplr__Publish_Start_DateTime__c": 1,
                            "Simpplr__Popularity_Score__c": 1,
                            "Simpplr__Site__r_Id": 1,
                            "Simpplr__Primary_Author__r_Id": 1,
                            "Simpplr__Pages_Category__r_Simpplr__Name__c": 1,
                            "Simpplr__Is_Deleted__c": 1,
                            "Simpplr__Type__c": 1
                        }
                    },
                    {
                        "collection_name": "Simpplr__Content__Interaction__c",
                        "output_key": "item_raw_df",
                        "query": {"org_id": "${org_id}", "LastModifiedDate": {"$$gte": str(train_from_timestamp)}},
                        "projection": {
                            "Simpplr__Content__Id": 1,
                            "Simpplr__People__Id": 1,
                            "Simpplr__View_Count__c": 1
                        }
                    },
                    {
                        "collection_name": "Simpplr__People__c",
                        "output_key": "people_raw_df",
                        "query": {"org_id": "${org_id}"},
                        "projection": {
                            "Simpplr__People__Id": 1,
                            "Simpplr__User__c": 1
                        }
                    },
                    {
                        "collection_name": "EntitySubscriptionPeople",
                        "output_key": "people_followed_raw_df",
                        "query": {"org_id": "${org_id}"},
                        "projection": {
                            "ParentId": 1,
                            "SubscriberId": 1
                        }
                    },
                    {
                        "collection_name": "Simpplr__Simpplr__Site__c",
                        "output_key": "site_raw_df",
                        "query": {"org_id": "${org_id}"},
                        "projection": {
                            "Id": 1,
                            "Simpplr__Chatter_Group_Id__c": 1,
                            "Simpplr__Site_Type__c": 1
                        }
                    },
                    {
                        "collection_name": "Simpplr__Carousel_Content__c",
                        "output_key": "carousel_content_df",
                        "query": {
                            "org_id": "${org_id}",
                            "Simpplr__Is_Deleted__c": "false"
                        },
                        "projection": {
                            "Simpplr__Order__c": 1,
                            "Simpplr__Simpplr_Content__c": 1,
                        }
                    },
                    {
                        "collection_name": "Simpplr__Must_Read_Audit__c",
                        "output_key": "must_read_audit_df",
                        "query": {
                            "org_id": "${org_id}",
                            "Simpplr__Is_Deleted__c": "false"
                        },
                        "projection": {
                            "Simpplr__Content__c": 1,
                            "Simpplr__Audience_Type__c": 1,
                            "Simpplr__Expiry_DateTime__c": 1,
                            "Simpplr__Mark_DateTime__c": 1,
                            "Simpplr__Removed_DateTime__c": 1
                        }
                    },
                    {
                        "collection_name": "Simpplr__Must_Read_Confirm_History__c",
                        "output_key": "must_read_conf_hist_df",
                        "query": {
                            "org_id": "${org_id}"
                        },
                        "projection": {
                            "Simpplr__Content__c": 1,
                            "Simpplr__People__c": 1,
                        }
                    },
                    {
                        "collection_name": "Simpplr__Site_Role__c",
                        "output_key": "site_role_df",
                        "query": {
                            "org_id": "${org_id}",
                            "Simpplr__Is_Deleted__c": "false"
                        },
                        "projection": {
                            "Simpplr__Site__c": 1,
                            "Simpplr__Is_Member__c": 1,
                            "Simpplr__People__c": 1
                        }
                    }
                ],
                "out_format": "df"
            },
            "data_harmonisation": {
                "func": "simpplr_data_harmonisation"
            },
            "user_data_prep": {
                "func": "simpplr_page_reco_data_prep",
                "content_id_key": "Simpplr__Content__Id",
                "people_id_key": "Simpplr__People__Id",
                "user_item_data_key": "item_raw_df",
                "content_data_key": "content_raw_df",
                "min_user_content": 5,
                "input_content_format": "df",
                "out_key": "user_item_raw_df"
            },
            "data_prep": {
                "func": "eapl_implicit_data_prep",
                "user_id_col": "Simpplr__People__Id",
                "item_id_col": "Simpplr__Content__Id",
                "events": "Simpplr__View_Count__c",
                "user_item_df_key": "user_item_raw_df"
            },
            "init_model": {
                "func": "eapl_implicit_init",
                "model_params": {
                    "factors": 63,
                    "iterations": 25,
                    "calculate_training_loss": True
                },
                "algo": "als",
                "model_key": "colab_model_${org_id}"
            },
            "train_model": {
                "func": "eapl_implicit_train_model",
                "model_key": "colab_model_${org_id}",
            },
            "gen_reco": {
                "func": "simpplr_gen_reco",
                "model_key": "colab_model_${org_id}",
                "index": "pr_${index}",
                "liked_item": True,
                "user_id_col": "Simpplr__People__Id",
                "item_id_col": "Simpplr__Content__Id",
                "sprs_matx_test_key": "sprs_matx_test",
                "top_n": 50,
                "min_user_content": 5,
                "days_to_subtract": 30,
                "days_to_subtract_pf_sf": 8,
                "model_score_threshold": 0,
                "out_key": "user_reco"
            },
            "index_reco_records": {
                "func": "simpplr_content_reco_indexing",
                "index": "pr_${index}",
                "text_key": "user_reco",
                "indexing_method": "index_bulk",
                "update_body": {
                    "query": {
                        "match_all": {}
                    },
                    "script": "ctx._source.record_status = 'inactive';"
                },
                "del_body": {
                    "query": {
                        "bool": {
                            "must_not": {
                                "term": {
                                    "record_status": "active"
                                }
                            }
                        }
                    }
                }
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": [
                ]
            }
        }
    },
    "page_reco_realtime": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "data_substitutions_check",
                "init_es",
                "get_reco",
                "email_reco_filtering",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "init_es": "from .elastic_search_custom import es_search_fmap",
                    "simpplr_email_reco_filtering": "from .simpplr_content_reco import simpplr_content_reco_fmap",
                    "mongodb_connect": "from .simpplr_mongodb_connection import eapl_simpplr_mogodb_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "eapl_implicit_data_prep": "from .reco_implicit import collab_implicit_fmap",
                }
            },
            "data_substitutions_check": {
                "func": "simpplr_basic_criteria_checks",
                "primarykeys": {"org_id": "str", "index": "str", "topn": "int", "mongodb_database_name": "str",
                                "mongodb_refresh": "bool"},
                "value_check_dict": {"topn": (0, 50)},
                "exc_emptykeys": ["topn", "mongodb_refresh"],
                "value_sync_list": [['org_id', 'index']],
                "text_key": "substitutions"
            },
            "init_es": simpplr_generic_cfgs["init_es"],
            "get_reco": {
                "func": "simpplr_get_reco",
                "index": "pr_${index}",
                "top_recom": "${topn}",
                "user_ids": "${reco_user_ids}",
                "reco_out_key": "content_recommendations",
            },
            "email_reco_filtering": {
                "func": "simpplr_email_reco_filtering",
                "mongodb_connection": simpplr_generic_cfgs["mongodb_connection"],
                "org_id": "${org_id}",
                "collection_name": "Simpplr__Content_Recommendation_Count__c",
                "email_reco_key": "email_reco_df",
                "input_key": "content_recommendations",
                "mongo_db_key": "mongodb",
                "output_key": "email_reco_df",
                "top_recom": "${topn}",
                "query": {
                    "org_id": "${org_id}",
                    "Simpplr__Content_Recommendation_Count__c": {"$$gte": 2},
                    "Simpplr__People__Id": {"$$in": "${reco_user_ids}"}
                },
                "projection": {
                    "Simpplr__Content__Id": 1,
                    "Simpplr__People__Id": 1,
                    "Simpplr__Content_Recommendation_Count__c": 1
                },
                "out_format": "df"
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": ["content_recommendations"]
            }
        }
    },
    "eapl_warm_up_simpplr": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": [
                "import_funcs",
                "init_use_model_st",
                "spacy_model_md",
                "spacy_model_sm",
                "start_redis",
                "hs_init_es_search",
                "hs_init_es_generic",
                "init_es",
                "mongodb_connection",
                "manage_data_keys"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "simpplr_extract_tags": "from .simpplr_tags_script import simpplr_pproc_fmap",
                    "eapl_semantic_drop_duplicate": "from .text_embed import use_embed_map",
                    "eapl_kpi_non_df_ops": "from .eapl_kpi_ref import eapl_kpi_fmap",
                    "entityruler_init": "from .nlp_ent_extraction import nlp_ent_extraction_fmap",
                    "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap",
                    "init_es": "from .elastic_search_custom import es_search_fmap",
                    "eapl_implicit_data_prep": "from .reco_implicit import collab_implicit_fmap",
                    "mongodb_connect": "from .simpplr_mongodb_connection import eapl_simpplr_mogodb_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "start_redis": "from .simpplr_redis_ops import simpplr_redis_ops_fmap"
                }
            },
            "init_use_model_st": simpplr_generic_cfgs["init_use_model_st"],
            "spacy_model_md": simpplr_generic_cfgs["init_nlp_md"],
            "spacy_model_sm": {
                "func": "eapl_nlp_pipeline_init",
                "model": "en_core_web_sm",
                "nlp_key": "nlp"
            },
            "start_redis": simpplr_generic_cfgs["start_redis"],
            "hs_init_es_search": simpplr_generic_cfgs["hs_init_es_search"],
            "hs_init_es_generic": simpplr_generic_cfgs["hs_init_es_generic"],
            "init_es": simpplr_generic_cfgs["init_es"],
            "mongodb_connection": simpplr_generic_cfgs["mongodb_connection"],
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": ["version"]
            }
        }
    },
    "simpplr_healthcheck": {
        "cfg_seq_type": "nlp_cfg_subs_template",
        "substitution_key": "substitutions",
        "cfg_subs_template": {
            "config_seq": ["import_funcs", "start_redis",
                           "hs_init_es_generic",
                           "mongodb_connection",
                           "manage_data_keys"],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "simpplr_extract_tags": "from .simpplr_tags_script import simpplr_pproc_fmap",
                    "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap",
                    "mongodb_connect": "from .simpplr_mongodb_connection import eapl_simpplr_mogodb_fmap",
                    "simpplr_basic_criteria_checks": "from .simpplr_utils import simpplr_utils_fmap",
                    "start_redis": "from .simpplr_redis_ops import simpplr_redis_ops_fmap"
                }
            },
            "start_redis": {**simpplr_generic_cfgs["start_redis"], **{"test_conn": True}},
            "hs_init_es_generic": simpplr_generic_cfgs["hs_init_es_generic"],
            "mongodb_connection": {**simpplr_generic_cfgs["mongodb_connection"],
                                   **{"test_conn": True}},
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": ["version"]
            }
        }
    }
}
