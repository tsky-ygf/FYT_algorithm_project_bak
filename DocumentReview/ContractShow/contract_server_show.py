#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 13:07
# @Author  : Adolf
# @Site    : 
# @File    : contract_server.py
# @Software: PyCharm
import time
import traceback
from typing import Dict

import requests
from flask import Flask, request
from flask_cors import CORS
import json
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from DocumentReview.ContractReview.showing_sample import BasicUIEAcknowledgementShow
import os

from Utils.http_response import response_successful_result, response_failed_result

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

CONTRACT_SERVER_DATA_PATH = "DocumentReview/ContractShow/contract_server_data.json"


def _get_support_contract_types():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("support_contract_types")


def _get_contract_type_list():
    support_contract_types = _get_support_contract_types()
    return [item.get("type_id") for item in support_contract_types]


contract_type_list = _get_contract_type_list()
acknowledgement_dict = {}
start_time = time.time()
for contract_type in contract_type_list:
    config_path = "DocumentReview/Config/{}.csv".format(contract_type)
    model_path = "model/uie_model/export_cpu/{}/inference".format(contract_type)
    acknowledgement_dict[contract_type] = BasicUIEAcknowledgementShow(config_path=config_path,
                                                                      model_path=model_path,
                                                                      device="cpu")
time_cost = time.time() - start_time
print("time_cost:{}".format(time_cost))


@app.route('/get_contract_type', methods=["get"])
def get_contract_type():
    support_contract_types = _get_support_contract_types()
    return response_successful_result(support_contract_types)


@app.route('/get_user_standpoint', methods=["get"])
def get_user_standpoint():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        user_standpoints = json.load(f).get("user_standpoints")
    return response_successful_result(user_standpoints)


# 创建线程执行器
executor = ThreadPoolExecutor(10)


@app.route('/get_contract_review_result', methods=["post"])
def get_contract_review_result():
    try:
        in_json = request.get_data()
        if in_json is not None:
            request_data = json.loads(in_json.decode("utf-8"))
            # 调用线程池执行器执行异步任务
            executor.submit(_async_task, request_data)
            return response_successful_result(None)
        else:
            return response_failed_result("no data")
    except Exception as e:
        logger.info(traceback.format_exc())
        return response_failed_result(traceback.format_exc())


def _async_task(request_data: Dict):
    unique_id = request_data['id']
    contract_type_id = request_data['contract_type_id']
    text = request_data['contract_content']
    usr = request_data['user_standpoint_id']

    if usr == 'party_a':
        usr = 'Part A'
    elif usr == 'part_b':
        usr = 'Part B'
    else:
        raise Exception("暂时不支持该用户立场")

    acknowledgement = acknowledgement_dict[contract_type_id]

    acknowledgement.review_main(content=text, mode="text", usr=usr)
    res = acknowledgement.review_result
    # 编排返回结果的内容
    result = []
    for review_point, review_result in res.items():
        result.append({
            "review_point": review_point,
            "show_name": review_result.get("show name") if review_result.get("show name") else review_point,
            "review_result": review_result.get("审核结果", ""),
            "review_content": review_result.get("内容", ""),
            "review_content_start": review_result.get("start", -1),
            "review_content_end": review_result.get("end", -1),
            "legal_advice": review_result.get("法律建议", ""),
            "legal_basis": review_result.get("法律依据", ""),
            "risk_level": review_result.get("风险等级", ""),
            "risk_point": review_result.get("风险点", "")
        })
    body = {"id": unique_id, "htZsList": result}
    requests.post(url="http://192.168.0.81:8092/serve/HtZs/notify", json=body)
    pass


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8110, debug=False)  # , use_reloader=False)
