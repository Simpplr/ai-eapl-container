import traceback
from django.http import HttpResponse, HttpResponseServerError, JsonResponse
from eapl_nlp.nlp.nlp_ops import eapl_data_process_fk_flow
from eapl_nlp.nlp.nlp_utils import eapl_config_import
from django.views.decorators.http import require_http_methods
import sys, json, os
import requests
import re
from django_rq import get_queue
from eapl_main.data_services import *

eapl_req_logger = logging.getLogger('eapl_ws_req')
eapl_err_logger = logging.getLogger('eapl_ws_err')


def custom_exception(nlp_cfg, e, info):
    status_code = 500
    response_dict = {}
    exc_type, exc_obj, exc_tb = info
    traceback_details = traceback.format_exc()

    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    if re.search("HTTP Error(.*?):", str(e)):
        status_code = int(re.search("HTTP Error(.*?):", str(e)).group(1))
    elif re.search("Error\(([1-5]\d{2})", str(e)):
        status_code = int(re.search("Error\(([1-5]\d{2})", str(e)).group(1))
    response_dict.update(
        {'status': 'fail', 'message': str(e), 'traceback': traceback_details, 'error_type': str(exc_type),
         'error_code': status_code})
    eapl_err_logger.error(
        {"nlp_cfg": nlp_cfg, "error": response_dict, "filename": fname, "line": exc_tb.tb_lineno})

    return status_code, response_dict


@require_http_methods(['GET'])
def eapl_flow_get(request, job_id: str = None):
    request_data = ({'job_id': job_id} if job_id else {}) or json.loads(request.body or '{}')
    eapl_req_logger.info(f'Got a GET Request')
    eapl_req_logger.info(f'Request: {request}')
    job_id = request_data.get('job_id', None)
    if job_id:
        return HttpResponse(content=job_status(request, 'default', job_id))
    else:
        return HttpResponse(content='ok')


@require_http_methods(['POST'])
def eapl_flow_post(request, api_name: str = None, version: str = None):
    if request.method == 'POST':
        try:
            request_data = json.loads(request.body or '{}')
            job_type = request_data.get('job_type', 'foreground')
            nlp_cfg = request_data.get('nlp_cfg', {})
            data = request_data.get('data', {})
            eapl_req_logger.info(f"Got a request...")
            eapl_req_logger.info(f"Running {job_type} job...")
            eapl_req_logger.info(f"nlp config: {nlp_cfg}")
            eapl_req_logger.info(f"data keys: {data.keys()}")
        except Exception as e:
            status, response_dict = custom_exception(nlp_cfg, e, sys.exc_info())
            return JsonResponse(response_dict, status=status)

        if job_type == 'foreground':
            try:
                response = main_call(request_data)
                response.update({"status": "success"})
            except Exception as e:
                status_code, response_dict = custom_exception(nlp_cfg, e, sys.exc_info())
                return JsonResponse(response_dict, status=status_code)

            return JsonResponse(response)

        elif job_type == 'background':
            try:
                q = get_queue(name='default', default_timeout=72000)
                job = q.enqueue(main_call_bg, request_data, description=f'Running Processing for config {nlp_cfg}')
            except Exception as e:
                status_code, response_dict = custom_exception(nlp_cfg, e, sys.exc_info())
                return JsonResponse(response_dict, status=status_code)
            else:
                cfg_seq = nlp_cfg.get("config_seq", "processing")
                return JsonResponse(
                    {'job_id': job.id, 'msg': f'Processing config for sequence {cfg_seq} in background.',
                     'status': 'success'})


@require_http_methods(['GET', 'POST'])
def eapl_flow(request):
    if request.method == 'POST':
        return eapl_flow_post(request)
    if request.method == 'GET':
        return eapl_flow_get(request)


def job_status(request, queue, jobid):
    q = get_queue(name=queue)
    j = q.fetch_job(job_id=jobid)

    if not j:
        return JsonResponse({'status': 'gone'})

    status = j.get_status()

    if status == "finished":
        return JsonResponse({'status': status, 'response': j.return_value})

    return JsonResponse({'status': status})


def main_call(request_data):
    output = dict()
    req_keys = list(request_data.keys())
    std_keys = ['nlp_cfg', 'data', 'output_keys']
    data = request_data.get('data', {})
    nlp_cfg = eapl_ws_get_nlp_cfg(request_data)

    for key in req_keys:
        if key not in std_keys:
            data.update({key: request_data[key]})

    data = eapl_data_process_fk_flow(data, nlp_cfg)

    output_keys = request_data.get('output_keys', list(data.keys()))
    for key in output_keys:
        output.update({key: data.get(key, None)})

    return output


def callback_req(request_data, status):
    if "callback_params" in request_data["data"]:
        requests.get(request_data['data']['callback_params'][status])


def main_call_bg(request_data):
    output = dict()
    try:
        output = main_call(request_data)
        callback_req(request_data, "success")
    except Exception as e:
        callback_req(request_data, "failure")
        raise

    return output


def eapl_ws_get_nlp_cfg(request_data):
    # Function to support multiple ways of getting nlp_cfg.
    # Currently dummy as support for redis is removed.
    nlp_cfg = request_data.get('nlp_cfg', {})
    return nlp_cfg


def test_eapl_ws():
    return None


if __name__ == '__main__':
    test_eapl_ws()
