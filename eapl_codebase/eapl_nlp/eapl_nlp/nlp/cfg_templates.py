nlp_cfg_templates = {

    'text_summarization_init_cfg':
        {
            'config_seq': ['import_funcs', 'init_pipe', 'eapl_txt_sum_init'],

            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_txt_sum_init": "from .text_sum import text_sum_fmap"
                }
            },

            'init_pipe': {
                'func': 'eapl_nlp_pipeline_init',
            },

            'eapl_txt_sum_init': {
                'func': 'eapl_txt_sum_init',
            }
        },

    'text_summarization_inference_config':
        {
            'config_seq': ['init_pipe', 'record_nlp_ops', "keep_keys"],
            'init_pipe': {
                'func': 'eapl_nlp_pipeline_init',
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
            "keep_keys": {
                "func": "manage_data_keys",
                "keep_keys": ['text_obj']
            }

        },

    "t5_base_question_gen_init":
        {
            "config_seq": ["import_t5", "init_pipe", "init_t5_pipe"],

            'init_pipe': {
                'func': 'eapl_nlp_pipeline_init',
            },
            "import_t5": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_t5_base_init": "from .t5_base_qg import t5_func_map"
                }
            },
            'init_t5_pipe': {
                'func': 'eapl_nlp_t5_base_pipeline_init',
                't5_obj': 't5'
            }
        },

    "t5_base_question_gen_inference":
        {
            "config_seq": ["init_t5_pipe", "record_nlp_ops", "keep_keys"],

            'init_t5_pipe': {
                'func': 'eapl_nlp_t5_base_pipeline_init',
                't5_obj': 't5'
            },
            "record_nlp_ops": {
                'func': 'eapl_nlp_record_process',
                'ops': [
                    {
                        'op': 'create_t5_base_question',
                        't5_obj': 't5',
                        'input_key': 'txt',
                        'out_key': 'questions',
                        'answer_style': 'sentences',
                        'qas': 'questions_rich',
                        'score_threshold': 2.0,
                        'num_questions': 100
                    }
                ]
            },
            "keep_keys": {
                "func": "manage_data_keys",
                "keep_keys": ['text_obj']
            }
        },

    "t5_pipe_question_gen_init":
        {
            "config_seq": ["import_t5", "init_pipe", "init_t5_pipe"],
            'init_pipe': {
                'func': 'eapl_nlp_pipeline_init',
            },
            "import_t5": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_t5_pipe_init": "from .t5_pipelines import t5_func_map"
                }
            },
            'init_t5_pipe': {
                'func': 'eapl_nlp_t5_pipeline_init',
                't5_obj': 't5'
            }
        },

    "t5_pipe_question_gen_inference":
        {
            'config_seq': ['init_t5_pipe', 'record_nlp_ops', "keep_keys"],

            'init_t5_pipe': {
                'func': 'eapl_nlp_t5_pipeline_init',
                't5_obj': 't5'
            },

            'record_nlp_ops': {
                'func': 'eapl_nlp_record_process',
                'ops': [
                    {
                        'op': 'create_t5_question',
                        't5_obj': 't5',
                        'txt': 'txt',
                        'questions': 'questions_para'
                    },
                ]
            },
            "keep_keys": {
                "func": "manage_data_keys",
                "keep_keys": ['text_obj']
            }
        },

    "alt_utt":
        {
            "config_seq": [
                "import", "eapl_alt_utt_init", "record_nlp_ops", "keep_keys"
            ],
            "import": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_alt_utt_init": "from .alt_utt import alt_utt_func_map"
                }
            },
            "eapl_alt_utt_init": {
                "func": "eapl_alt_utt_init"
            },
            "record_nlp_ops": {
                "func": "eapl_nlp_record_process",
                "ops": [
                    {
                        "op": "eapl_alt_utt_rec",
                        "input_key": "txt",
                        "output_key": "questions",
                        "max_utt": 2
                    }
                ]
            },
            "keep_keys": {
                "func": "manage_data_keys",
                "keep_keys": ['text_obj']
            }
        },

    "text_embed":
        {
            'config_seq': ['import', 'init_use', 'record_use_embed', "keep_keys"],
            'import': {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_txt_embed_init": "from .text_embed import use_embed_map"
                }
            },
            'init_use': {
                'func': 'eapl_use_init',
                'model_url': 'https://tfhub.dev/google/universal-sentence-encoder/4',
                'model_key': 'use_model'
            },
            'record_use_embed': {
                'func': 'eapl_nlp_record_process',
                'ops': [
                    {
                        'op': 'eapl_semantic_cond_match',
                        'x_params': {'input_key': 'questions_rich', 'txt_key': 'question', 'vec_key': None},
                        'y_params': {'input_key': 'data_drop_list', 'txt_key': 'txt', 'vec_key': None, "td_flag": False,
                                     "dtype": "list"},
                        'filter_cond': "dist < 0.9",
                        'model_key': 'use_model'
                    },
                    {
                        'op': 'eapl_semantic_drop_duplicate',
                        'text_data_type': 'list_of_dicts',
                        'model_key': 'use_model',
                        'text_key': 'questions_rich',
                        'input_key': "question",
                        'out_key': 'semantic_drop_duplic_output'
                    }
                ]
            },
            "keep_keys": {
                "func": "manage_data_keys",
                "keep_keys": ['text_obj']
            }
        },

    "spacy_classifier":
        {
            'config_seq': ["import", 'init_pipe', 'init_pipe_clfr', 'train_model', 'save_model', 'load_model',
                           'records_label_scoring', "keep_keys"],
            "import": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_spacy_classifer_init": "from .spacy_classifier import eapl_spacy_classifier_func_map",
                }
            },
            'init_pipe': {
                'func': 'eapl_nlp_pipeline_init',
                'nlp_key': 'spacy_init'
            },
            'init_pipe_clfr': {
                'func': 'eapl_spacy_classifer_init',
                'nlp_key': 'spacy_init',
                'exclusive_classes': False
            },

            'train_model': {
                'func': 'train_model_spacy',
                'train_data_fmt': 'records',
                'input_data_key': 'text_obj',
                'input_key': 'txt',
                'label_key': 'label',
                'nlp_key': 'spacy_init',
                'epoch': 10,
                'pipe_name': 'textcat'
            },

            'save_model': {
                'func': 'eapl_spacy_save_nlp',
                'nlp_key': 'spacy_init',
                'output_dir': 'spacy_tmp'

            },
            'load_model': {
                'func': 'eapl_nlp_pipeline_init',
                'model': 'spacy_ques_check',
                'nlp_key': 'spacy_init',
            },

            'records_label_scoring': {
                'func': 'eapl_nlp_record_process',
                'text_obj': 'test_obj',
                'ops': [
                    {
                        "op": "create_spacy_doc",
                        'nlp_key': 'spacy_init',
                        'txt_key': 'body',
                        'doc_key': 'test_doc_key'
                    },
                    {
                        "op": "eapl_spacy_inference",
                        'doc_key': 'test_doc_key',
                        'out_key': 'label',
                    },
                    {
                        "op": "manage_text_dict_keys",
                        "pop_keys": ['test_doc_key']
                    },
                ]
            },
            "keep_keys": {
                "func": "manage_data_keys",
                "keep_keys": ['text_obj']
            }
        },

    'hs_index_data':
        {
            "config_seq": [
                "import_funcs",
                "hs_init_faiss_search",
                "load_prod_data",
                "hs_create_index_data",
                "write_documents",
                "save_hs_data2glob",
                "handle_json_error"
            ],
            "import_funcs": {
                "func": "eapl_config_import",
                "imports": {
                    "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap"
                }
            },
            "hs_init_faiss_search": {
                "func": "eapl_hs_init_setup",
                "hs_setup_pipeline": [
                    {
                        "func": "eapl_hs_docstore",
                        "docstore_type": "FAISSDocumentStore",
                        "docstore_filepath": "faiss_index_sapio.faiss",
                        "docstore_params": {
                            "sql_url": "sqlite:///sem.db?check_same_thread=False",
                            "vector_dim": 768,
                            "update_existing_documents": True
                        },
                        "delete_all_docs_flag": True
                    },
                    {
                        "func": "eapl_hs_retriever",
                        "retriever_type": "EmbeddingRetriever",
                        "retriever_params": {
                            "embedding_model": "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                            "model_format": "sentence_transformers",
                            "use_gpu": False
                        }
                    },
                    {
                        "func": "eapl_hs_create_pipe",
                        "pipe_type": "DocumentSearchPipeline"
                    }
                ]
            },
            "load_prod_data": {
                "func": "eapl_read_file_data",
                "read_method": "pd_read_csv",
                "method_params": {
                    "nrows": 10,
                    "filepath_or_buffer": "https://s3.amazonaws.com/emplay.botresources/sap+io/sap_io_qg.csv"
                },
                "refresh": True,
                "out_key": "ref_df"
            },
            "hs_create_index_data": {
                "func": "eapl_hs_df2docs",
                "df_key": "ref_df",
                "docs_key": "docs",
                "text_fields": [
                    "question"
                ],
                "meta_data_fields": ["Title Hierarchy"]
            },
            "write_documents": {
                "func": "eapl_hs_write_documents",
                "update_embeddings": True,
                "hs_obj_key": "hs_obj",
                "docs_key": "docs"
            },
            "save_hs_data2glob": {
                "func": "eapl_hs_obj_mgmt",
                "transfer": "data_to_glob",
                "hs_obj_key": "hs_obj"
            },
            "handle_json_error": {
                "func": "eapl_handle_json_ser_err"
            },
            "manage_data_keys": {
                "func": "manage_data_keys",
                "keep_keys": [
                    "bot_name"
                ]
            }
        },

    'hs_setup': {
        "config_seq": [
            "import_funcs",
            "hs_init_faiss_search",
            "hs_pipe_run",
            "manage_dict_keys",
            "manage_data_keys",
            "handle_json_error"
        ],
        "import_funcs": {
            "func": "eapl_config_import",
            "imports": {
                "eapl_hs_obj_mgmt": "from .eapl_haystack_search import eapl_haystack_search_fmap"
            }
        },
        "hs_init_faiss_search": {
            "func": "eapl_hs_init_setup",
            "refresh": False,
            "hs_setup_pipeline": [
                {
                    "func": "eapl_hs_docstore",
                    "docstore_type": "FAISSDocumentStore",
                    "docstore_params": {
                        "sql_url": "sqlite:///sem.db?check_same_thread=False"
                    },
                    "docstore_load_params": {
                        "faiss_file_path": "faiss_index_sapio.faiss"
                    }
                },
                {
                    "func": "eapl_hs_retriever",
                    "retriever_type": "EmbeddingRetriever",
                    "retriever_params": {
                        "embedding_model": "sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                        "model_format": "sentence_transformers",
                        "use_gpu": False
                    }
                },
                {
                    "func": "eapl_hs_create_pipe",
                    "pipe_type": "DocumentSearchPipeline"
                }
            ]
        },
        "load_hs_glob2data": {
            "func": "eapl_hs_obj_mgmt",
            "transfer": "glob_to_data",
            "hs_obj_key": "hs_obj"
        },
        "hs_pipe_run": {
            "func": "eapl_hs_pipe_run",
            "query_key": "query",
            "results_key": "search_results",
            "pipe_params": {
                "top_k_retriever": 10
            }
        },
        "manage_dict_keys": {
            "func": "eapl_nlp_record_process",
            "text_obj": "search_results",
            "ops": [
                {
                    "op": "manage_text_dict_keys",
                    "pop_keys": [
                        "score"
                    ]
                }
            ]
        },
        "manage_data_keys": {
            "func": "manage_data_keys",
            "keep_keys": [
                "search_results"
            ]
        },
        "handle_json_error": {
            "func": "eapl_handle_json_ser_err"
        }
    }

}
