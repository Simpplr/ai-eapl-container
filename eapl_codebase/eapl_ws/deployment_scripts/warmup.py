import requests
import json
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filepath", required=True, help="JSON Configs URL path")
    args = vars(ap.parse_args())
    filepath = args['filepath']

    url = "http://127.0.0.1/eapl/eapl-nlp/"
    f = requests.get(filepath)
    payload_lst = json.loads(f.text)
    headers = {
        'Authorization': 'Basic c2ltcHBscjpxd2VydHlAMTIz',
        'Content-Type': 'application/json'
    }
    for p in payload_lst:
        # print(p)
        response = requests.request("POST", url, headers=headers, data=json.dumps(p))
        print(response.text)
