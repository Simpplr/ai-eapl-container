import os
import wget
import sys
from urllib.parse import urlparse
import shutil
import boto3
from botocore.exceptions import NoCredentialsError
from contextlib import contextmanager
import logging
import zipfile
import tarfile
import tempfile
from requests.auth import HTTPBasicAuth
import json
import pandas as pd
import importlib

try:
    from .nlp_glob import *
    from .nlp_ops import eapl_data_process_fk_flow
except ImportError:
    from nlp_glob import *
    from nlp_ops import eapl_data_process_fk_flow

nlp_utils_logger = logging.getLogger(__name__)


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def wget_with_temp_download(src_path, dest_path):
    if os.path.exists(dest_path) and os.path.isfile(dest_path):
        nlp_utils_logger.debug(f"Deleting the file: {dest_path}")
        os.remove(dest_path)
    dest_abspath = os.path.abspath(dest_path)
    with tempfile.TemporaryDirectory() as td:
        with cwd(td):
            wget.download(src_path, dest_abspath)


def eapl_copy_file(data, cfg):
    src_path = cfg.get('src_path')
    dest_path = cfg.get('dest_path')
    url_scheme = cfg.get('url_scheme', ('http', 'https', 's3'))

    is_url = urlparse(src_path).scheme in url_scheme
    if is_url:
        wget_with_temp_download(src_path, dest_path)
    else:
        shutil.copy(src_path, dest_path)
    return data


def eapl_uc_file(data, cfg):
    file_path = cfg["file_path"]
    out_path = cfg["out_path"]
    extension = file_path.split(".")[-1]
    if extension == 'zip':
        z = zipfile.ZipFile(file_path)
        z.extractall(out_path)
    elif extension == 'tar' or extension == 'gz':
        z = tarfile.TarFile(file_path)
        z.extractall(out_path)
    return data


def eapl_modify_config(data, cfg):
    input_cfg_key = cfg['input_cfg_key']
    upd_cfg_key = cfg['upd_cfg_key']
    output_cfg_key = cfg.get('output_cfg_key', input_cfg_key)
    pos = cfg.get('pos', None)  # Insert at the end of config_seq by default
    input_cfg = data.get(input_cfg_key, {}).copy()
    upd_cfg = data[upd_cfg_key].copy()

    upd_cfg_seq = upd_cfg.pop('config_seq')
    input_cfg_seq = input_cfg['config_seq'].copy()
    if pos is None:
        pos = len(input_cfg_seq)
    input_cfg_seq[pos:pos] = upd_cfg_seq
    input_cfg['config_seq'] = input_cfg_seq
    input_cfg.update(upd_cfg)

    data[output_cfg_key] = input_cfg
    return data


def eapl_download_file(data, cfg):
    f_name_or_data = cfg['file']
    dest_name = cfg.get('dest_name', ".")
    filemode = cfg.get('filemode', 'fpath')
    dest_path = f"{dest_name}"
    if 'url' == filemode:
        nlp_utils_logger.debug(f"Downloading file {f_name_or_data} to {dest_path}")
        wget.download(f_name_or_data, dest_path)
    elif 'fpath' == filemode:
        nlp_utils_logger.debug(f"Copying file {f_name_or_data} to {dest_path}")
        shutil.copy(f_name_or_data, dest_path)
    elif 'txt' == filemode:
        nlp_utils_logger.debug(f"Copying file text stream data to {dest_path}")
        fp = open(dest_path, "w")
        fp.write(f_name_or_data)
        fp.close()
    else:
        nlp_utils_logger.debug(f"Something incorrect with passed file params : {cfg}")

    return data


def eapl_download_import_file(data, cfg):
    tdir = cfg.get('tdir')
    file_url_list = cfg.get('file_info')
    refresh = cfg.get('refresh', False)
    tdir_local = False
    if tdir:
        if not os.path.exists(tdir):
            os.makedirs(tdir, exist_ok=True)
    else:
        tdir = tempfile.mkdtemp()
        nlp_utils_logger.debug(f"Temp directory: {tdir}")
        tdir_local = True

    with cwd(tdir):
        try:
            for file_obj in file_url_list:
                func_key = file_obj.get('func_key', None)
                if func_key not in nlp_func_map.keys() or refresh:
                    f_name_or_data = file_obj.get('file')
                    dest_name = cfg.get('dest_name', ".")
                    filemode = file_obj.get('filemode', 'fpath')
                    if filemode == 'datakey':
                        src_key = file_obj.get('src_key', 'src_key')
                        comma_char = file_obj.get('comma_char', None)
                        f_name_or_data = eapl_extract_nested_dict_value(data, src_key)
                        if comma_char:
                            f_name_or_data = f_name_or_data.replace(comma_char, ",")

                    import_stmt = file_obj.get("import_stmt", None)
                    dest_path = f"{dest_name}"
                    if 'url' == filemode:
                        nlp_utils_logger.debug(f"Downloading file {file_obj} to {dest_path}")
                        wget.download(f_name_or_data, dest_path)
                    elif 'fpath' == filemode:
                        nlp_utils_logger.debug(f"Copying file {file_obj} to {dest_path}")
                        shutil.copy(f_name_or_data, dest_path)
                    elif filemode in ['txt', 'datakey']:
                        nlp_utils_logger.debug(f"Copying file text stream data to {dest_path}")
                        fp = open(dest_path, "w")
                        fp.write(f_name_or_data)
                        fp.close()
                    else:
                        nlp_utils_logger.debug(f"Something incorrect with passed file params : {file_obj}")

                    if import_stmt:
                        import_mod_name = import_stmt.split()[:2][-1]  # process to get module name
                        sys.path.append(tdir)
                        if import_mod_name in sys.modules:
                            reload_module = sys.modules[import_mod_name]
                            importlib.reload(reload_module)
                        else:
                            importlib.import_module(import_mod_name)

        except Exception as e:
            nlp_utils_logger.debug(f"Exception: {e}")

    if tdir_local:
        shutil.rmtree(tdir)

    return data


def eapl_config_import(data, cfg):
    import_dct = cfg['imports']
    for func_key in import_dct.keys():
        try:
            if func_key not in nlp_func_map.keys():
                import_stmt = import_dct[func_key]
                import_mod_name = import_stmt.split()[:2][-1]  # process to get module name & backward compatibility
                import_mod_name = import_mod_name.lstrip(".")
                try:
                    importlib.import_module(f".{import_mod_name}", package="eapl_nlp.nlp")
                except ImportError:
                    importlib.import_module(import_mod_name)
                nlp_utils_logger.debug(f"Import of statement: {import_stmt} successful")
        except Exception as e:
            nlp_utils_logger.debug(f"Exception: {e}")
            nlp_utils_logger.debug(f"Import of statement: {import_dct[func_key]} failed")

    return data


def eapl_fileio_aws(data, cfg):
    local_file = cfg['local_file']
    bucket = cfg['bucket']
    s3_file = cfg['s3_file']
    access_key = cfg.get('access_key', None)
    secret_key = cfg.get('secret_key', None)
    upload = cfg.get('upload', False)

    if access_key and secret_key:
        s3 = boto3.client('s3', aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key)
    else:
        s3 = boto3.client('s3')

    try:
        if upload:
            s3.upload_file(local_file, bucket, s3_file)
            nlp_utils_logger.debug("Upload of {local_file} to AWS Successful")
        else:
            s3.download_file(bucket, s3_file, local_file)
            nlp_utils_logger.debug("Download of {s3_file} from AWS Successful")
    except FileNotFoundError:
        nlp_utils_logger.debug("The file was not found")
    except NoCredentialsError:
        nlp_utils_logger.debug("Credentials not available")

    return data


def eapl_read_file_data(data, cfg):
    read_method = cfg.get('read_method', 'pd_read_csv')
    refresh = cfg.get('refresh', False)
    file_or_url_path = cfg.get('file_or_url_path', None)
    out_key = cfg['out_key']
    method_params = cfg.get('method_params', {}).copy()  # For controlling parameters of read methods

    if out_key not in nlp_glob or refresh:
        nlp_utils_logger.debug(
            f"eapl_read_file_data out_key: {out_key} | Refresh Flag: {refresh}. Loading from path: {file_or_url_path}")
        if read_method == 'pd_read_csv':
            if file_or_url_path:
                method_params.update({'filepath_or_buffer': file_or_url_path})
            df = pd.read_csv(**method_params)
            nlp_glob[out_key] = df
        else:
            nlp_utils_logger.debug(f"read_method not supported")

    data[out_key] = nlp_glob[out_key]
    return data


def eapl_handle_json_ser_err(data, cfg):
    def_json_params = {
        'default': lambda o: '<not serializable>'
    }
    json_params = cfg.get('json_params', {}).copy()
    json_params.update(def_json_params)
    nlp_utils_logger.debug(f"")
    json_str = json.dumps(data, **json_params)
    data = json.loads(json_str)
    return data


def eapl_nlp_ws(data, cfg):
    url = cfg.get('eapl_ws_url', "http://127.0.0.1:8000/eapl/eapl-nlp/")
    nlp_cfg_src = cfg.get("nlp_cfg_src", None)
    eapl_ws_error = cfg.get("eapl_ws_error", "eapl_ws_error")
    data_keys = cfg.get("data_keys", data.keys())
    auth = cfg.get("auth", None)
    if auth:
        username, pwd = auth
    if nlp_cfg_src == "data":
        nlp_cfg_key = cfg["nlp_cfg_key"]
        nlp_cfg = data[nlp_cfg_key]
    else:
        nlp_cfg = cfg["nlp_cfg"]

    data_ws = dict()
    for key in data_keys:
        data_ws.update({key: data[key]})
    req_body = {
        "nlp_cfg": nlp_cfg,
        "data": data_ws,
    }
    nlp_utils_logger.debug(f"nlp_cfg: {nlp_cfg}")
    if auth:
        r = requests.post(url=url, json=req_body, auth=HTTPBasicAuth(username, pwd))
    else:
        r = requests.post(url=url, json=req_body)
    try:
        ws_resp_dct = json.loads(r.content)
        data.update(ws_resp_dct)
    except Exception as e:
        nlp_utils_logger.debug(f"Exception: {e}")
        err_string = f"eapl_nlp_ws Error : {r.status_code} || {r.content} || {cfg}"
        data[eapl_ws_error] = err_string
    return data


nlp_utils_fmap = {
    'eapl_config_import': eapl_config_import,
    'eapl_copy_file': eapl_copy_file,
    'eapl_uc_file': eapl_uc_file,
    'eapl_fileio_aws': eapl_fileio_aws,
    'eapl_modify_config': eapl_modify_config,
    'eapl_download_import_file': eapl_download_import_file,
    'eapl_download_file': eapl_download_file,
    'eapl_handle_json_ser_err': eapl_handle_json_ser_err,
    'eapl_nlp_ws': eapl_nlp_ws,
    'eapl_read_file_data': eapl_read_file_data
}
nlp_func_map.update(nlp_utils_fmap)


def test_nlp_utils():
    from pprint import pprint
    nlp_cfg = {
        'config_seq': ['copy_test', 'import_test', 'aws_upload', 'aws_download'],
        'copy_test': {
            'func': 'eapl_copy_file',
            'src_path': 'https://www.w3.org/TR/PNG/iso_8859-1.txt',
            'dest_path': 'qg_inp_d_aws.json'
        },

        'import_test': {
            'func': 'eapl_config_import',
            'imports': {
                'eapl_txt_sum_init': "from text_sum import text_sum_fmap"
            }
        },

        'aws_upload': {
            'func': 'eapl_fileio_aws',
            'local_file': 'qg_inp.json',
            'bucket': 'test-aws-tr',
            's3_file': 'qg_inp.json',
            'upload': True
        },

        'aws_download': {
            'func': 'eapl_fileio_aws',
            'local_file': 'qg_inp_d_aws.json',
            'bucket': 'test-aws-tr',
            's3_file': 'qg_inp.json',
            'upload': False
        }
    }

    data = dict()
    func = nlp_func_map['eapl_data_process_fk_flow']
    _ = func(data, nlp_cfg)
    pprint(nlp_func_map.keys())

    data = {
        'inp_cfg': nlp_cfg,
        'upd_cfg': {
            'config_seq': ['a', 'b'],
            'a': {
                'func': 'eapl_copy_file_a',
            },

            'b': {
                'func': 'eapl_config_import_b',
            },
        }
    }
    cfg = {
        'input_cfg_key': 'inp_cfg',
        'output_cfg_key': 'out_cfg',
        'upd_cfg_key': 'upd_cfg',
    }
    data = eapl_modify_config(data, cfg)

    return None


if __name__ == '__main__':
    test_nlp_utils()
