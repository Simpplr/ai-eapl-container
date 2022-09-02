from eapl.df.eapl_kpi import *

try:
    from .nlp_glob import nlp_func_map
    from .nlp_ops import *
except ImportError:
    from nlp_glob import nlp_func_map
    from nlp_ops import *


def eapl_data_fmt_conv(data, cfg):
    input_key = cfg['input_key']
    output_key = cfg['output_key']
    format_conv = cfg.get('format_conv', 'records_to_df')

    inp_obj = data[input_key]
    if format_conv == 'records_to_df':
        output = pd.DataFrame.from_records(inp_obj)
    elif format_conv == 'df_to_records':
        output = inp_obj.to_dict(orient='records')
    else:
        eapl_kpi_logger.debug(f"Unsupported format conversion: {format_conv}")

    data[output_key] = output
    return data


def eapl_kpi_non_df_ops(data, cfg):
    kpi_cfg = cfg.copy()
    df_func = kpi_cfg['df_func']

    kpi_cfg['format_conv'] = cfg.get('inp_format_conv', 'records_to_df')
    func = nlp_func_map['eapl_data_fmt_conv']
    data = func(data, kpi_cfg)

    kpi_cfg['input_df'] = kpi_cfg['output_df'] = cfg['output_key']
    func = nlp_func_map[df_func]
    data = func(data, kpi_cfg)

    kpi_cfg['format_conv'] = cfg.get('out_format_conv', 'df_to_records')
    kpi_cfg['input_key'] = kpi_cfg['output_df']
    func = nlp_func_map['eapl_data_fmt_conv']
    data = func(data, kpi_cfg)

    return data


eapl_kpi_fmap = {
    'eapl_col_value_counts': eapl_col_value_counts,
    'eapl_attribute_kpi_summary': eapl_attribute_kpi_summary,
    'eapl_datasize_summary_wrapper': eapl_datasize_summary_wrapper,
    'eapl_data_summary_wrapper': eapl_data_summary_wrapper,
    'eapl_groupby_data_summary': eapl_groupby_data_summary,
    'eapl_derived_cols_wrapper': eapl_derived_cols_wrapper,
    'eapl_df_agg_pct_kpis_wrapper': eapl_df_agg_pct_kpis_wrapper,
    'eapl_data_process_flow': eapl_data_process_flow,
    'eapl_save_data': eapl_save_data,
    'eapl_read_data': eapl_read_data,
    'eapl_data_quality_config': eapl_data_quality_config,
    'eapl_gpby_correl': eapl_gpby_correl,
    'eapl_data_fmt_conv': eapl_data_fmt_conv,
    'eapl_kpi_non_df_ops': eapl_kpi_non_df_ops
}
nlp_func_map.update(eapl_kpi_fmap)


def testeapl_kpi_config():
    from pprint import pprint
    csv_file_path = 'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/titanic_train.csv'
    nlp_cfg = {
        'config_seq': ['eapl_kpi_non_df_ops_cfg', 'convert_recs_to_df', 'modify_df', 'convert_df_to_recs'],

        'eapl_kpi_non_df_ops_cfg': {
            'func': 'eapl_kpi_non_df_ops',
            'df_func': 'eapl_derived_cols_wrapper',
            'input_key': 'inp_recs',
            'output_key': 'out_recs_e2e',
            'ops': [
                ['sort_values', ('Fare', False, None)]
            ]
        },

        'convert_recs_to_df': {
            'func': 'eapl_data_fmt_conv',
            'input_key': 'inp_recs',
            'output_key': 'idf',
            'format_conv': 'records_to_df'
        },

        'modify_df': {
            'func': 'eapl_df_agg_pct_kpis_wrapper',
            'input_df': 'idf',
            'output_df': 'odf',
            'gpbycols': ['Pclass'],

            'ops': [
                ['agg_feats', ('PassengerId', 'count', 'Pclass_count')]
            ]
        },

        'convert_df_to_recs': {
            'func': 'eapl_data_fmt_conv',
            'input_key': 'odf',
            'output_key': 'out_recs',
            'format_conv': 'df_to_records'
        },
    }

    data = {
        'inp_recs': pd.read_csv(csv_file_path).to_dict(orient='records')
    }

    func = nlp_func_map['eapl_data_process_fk_flow']
    data = func(data, nlp_cfg)
    pprint(data['out_recs_e2e'])

    return None


if __name__ == '__main__':
    testeapl_kpi_config()
