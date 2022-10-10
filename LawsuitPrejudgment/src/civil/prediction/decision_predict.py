# -*- coding: utf-8 -*-
import time
import requests
import json
import logging
import traceback
import re
import numpy as np

# http = 'http://dev.decision-apis.ii-ai.tech/models/invoke' # develop env

http = 'http://qa.decision-apis.ii-ai.tech/models/invoke'  # qa env
model_ids = {
    'ml': "58611f3a1190421fb08b2a518cbbaea5",
    'fasttext': "4d99820144a941aa866b0dc9703c8ed8",
    'textcnn': "54eb0a9f84dc45aebd8282d62d3753a4",
}

# http = 'http://decision-apis.ii-ai.tech/models/invoke'
# model_ids = {
#     'fasttext': "e32e88f3975941709a19544c13bb5445",
#     'textcnn': "145e9ab0f352466099bba57844ef6caa",
# }


def support_possibility_ml(problem, suqiu_type, fact):
    print("suqiu_type:", suqiu_type, ";fact:", fact)
    try:
        time1 = time.time()
        parameter = {"modelId": model_ids['ml'], "data": {"chaming_fact": fact, "new_suqiu": suqiu_type, "new_problem": problem}}  # develop env id:"d97d247bfdc24b53a0d130497c23917e"
        r = requests.post(http, json=parameter, timeout=2)
        result = r.text
        print(result)
        result = json.loads(result)
        time2 = time.time()
        p = result['data']['result']
        logging.info("time spent: %s; type of result: %s" % ((time2 - time1), type(result)))
        return p
    except:
        traceback.print_exc()
        logging.info("support possibility interface failed")
        return None


def support_possibility_fasttext(problem, suqiu_type, fact):
    print("suqiu_type:", suqiu_type, ";fact:", fact)
    try:
        time1 = time.time()
        parameter = {"modelId": model_ids['fasttext'], "data": {"chaming_fact": fact}}
        r = requests.post(http, json=parameter, timeout=1)
        result = r.text
        print(result)
        result = json.loads(result)
        time2 = time.time()
        p = result['data']['result_with_prob'][1][1]
        logging.info("time spent: %s; type of result: %s" % ((time2 - time1), type(result)))
        return p
    except:
        traceback.print_exc()
        logging.info("support possibility interface failed")
        return None


def support_possibility_textcnn(problem, suqiu_type, fact):
    print("suqiu_type:", suqiu_type, ";fact:", fact)
    try:
        time1 = time.time()
        parameter = {"modelId": model_ids['textcnn'], "data": {"chaming_fact": fact}}
        r = requests.post(http, json=parameter, timeout=1)
        result = r.text
        print(result)
        result = json.loads(result)
        time2 = time.time()
        p = result['data']['result_with_prob'][1][1]
        logging.info("time spent: %s; type of result: %s" % ((time2 - time1), type(result)))
        return p
    except:
        traceback.print_exc()
        logging.info("support possibility interface failed")
        return None


if __name__ == '__main__':
    problem = "婚姻家庭"
    suqiu_type = "离婚"
    fact = "双方自愿离婚。"
    result = support_possibility_fasttext(problem, suqiu_type, fact)
    print("result:", result)
