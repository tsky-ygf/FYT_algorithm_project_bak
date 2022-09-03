import json
import requests

from Utils import print_run_time

@print_run_time
def req_case(url_search_case, input_json):
    r = requests.post(url_search_case, json=input_json) #
    res = r.json()
    print(res)
    pass
if __name__=='__main__':
    url_search_case = 'http://172.19.82.199:8801/search_cases'
    query = "离婚"
    filter_conditions = {
        'type_of_case': ['故意'],
        'court_level': ['中级'],
        'type_of_document': ['判决'],
        'region': ['江苏'],
        'size': 10,
    }
    input_json = {
        "query": query
        , "filter_conditions": filter_conditions  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
    }
    req_case(url_search_case, input_json)
    # req_situa(url_situation, input_json)

