import json
import requests

from Utils import print_run_time

@print_run_time
def req_conditions(url_filter_conditions):
    r = requests.get(url_filter_conditions) #
    res = r.json()
    print(res)

@print_run_time
def req_case(url_search_case, input_json):
    r = requests.post(url_search_case, json=input_json) #
    res = r.json()
    print(res)
    pass

@print_run_time
def req_law_document(url_law_document, input_json_doc):
    r = requests.get(url_law_document, json=input_json_doc) #
    res = r.json()
    print(res)
    pass

if __name__=='__main__':
    # url_filter_conditions = 'http://172.19.82.199:8156/get_filter_conditions_of_case'
    url_search_case = 'http://172.19.82.199:8140/search_cases'
    # url_law_document = 'http://172.19.82.199:8156/get_law_document'
    query = "拐卖"
    filter_conditions = {
        'type_of_case': ['民事'],
        # 'court_level': ['中级'],
        # 'type_of_document': ['裁定书'],
        # 'region': ['青海省'],
        # 'size': 10,
    }
    input_json = {
        "page_number":1,
        "page_size":10,
        "query": query
        , "filter_conditions": filter_conditions  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
    }
    input_json_doc = {"doc_id": "7bfd4c29-ab17-4a61-a905-a9b2017ef685"}
    # req_conditions(url_filter_conditions)
    req_case(url_search_case, input_json)
    # req_law_document(url_law_document, input_json_doc)
    # req_situa(url_situation, input_json)