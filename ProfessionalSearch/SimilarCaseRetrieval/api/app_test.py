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
    url_search_case = 'http://172.19.82.199:8160/search_cases'
    # url_law_document = 'http://101.69.229.138:7145/get_law_document'
    query = "买卖"
    filter_conditions = {
        'type_of_case': ['执行'],
        'court_level': ['基层'],
        'type_of_document': ['判决'],
        'region': ['山东省'],
        # 'size': 10,
    }
    input_json = {
        "page_number":1,
        "page_size":10,
        "query": query
        , "filter_conditions": filter_conditions  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
    }
    # input_json_doc = {"doc_id": "66ed0e28-7d94-423a-8583-aae301134b1e"}
    # req_conditions(url_filter_conditions)
    req_case(url_search_case, input_json)
    # req_law_document(url_law_document, input_json_doc)
    # req_situa(url_situation, input_json)