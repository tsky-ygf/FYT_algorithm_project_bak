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
    # res = r.json()
    print(r)
    pass

@print_run_time
def req_law_document(url_law_document, input_json_doc):
    r = requests.get(url_law_document, json=input_json_doc) #
    res = r.json()
    print(res)
    pass

if __name__=='__main__':
    url_filter_conditions = 'http://101.69.229.138:8140/get_filter_conditions_of_case'
    url_search_case = 'http://101.69.229.138:8140/search_cases'
    url_law_document = 'http://101.69.229.138:8140/get_law_document'
    query = ""
    filter_conditions = {
        # 'type_of_case': ['刑事案件'],
        # 'court_level': ['基层'],
        # 'type_of_document': ['判决'],
        # 'region': ['江苏'],
        # 'size': 10,
    }
    input_json = {
        "query": query
        , "filter_conditions": filter_conditions  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
    }
    input_json_doc = {"doc_id": "24dbed45-904d-4992-aea7-a82000320181"}
    # req_conditions(url_filter_conditions)
    req_case(url_search_case, input_json)
    # req_law_document(url_law_document, input_json_doc)
    # req_situa(url_situation, input_json)

