#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 13:07
# @Author  : Adolf
# @Site    : 
# @File    : contract_server.py
# @Software: PyCharm
import traceback
from flask import Flask, request
from flask_cors import CORS
import json
from loguru import logger

from DocumentReview.ContractReview.basic_contract import BasicUIEAcknowledgement
import os

from Utils.http_response import response_successful_result

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

CONTRACT_SERVER_DATA_PATH = "DocumentReview/ContractShow/contract_server_data.json"

contract_type_list = ['fangwuzulin', 'jiekuan', 'jietiao', 'laodong', 'laowu', 'maimai']
acknowledgement_dict = {}
for contract_type in contract_type_list:
    config_path = "DocumentReview/Config/{}.csv".format(contract_type)
    model_path = "model/uie_model/new/{}/model_best/".format(contract_type)
    acknowledgement_dict[contract_type] = BasicUIEAcknowledgement(config_path=config_path,
                                                                  model_path=model_path,
                                                                  device_id=-1)


@app.route('/get_contract_type', methods=["get"])
def get_contract_type():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        support_contract_types = json.load(f).get("support_contract_types")
    return response_successful_result(support_contract_types)


@app.route('/get_user_standpoint', methods=["get"])
def get_user_standpoint():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        user_standpoints = json.load(f).get("user_standpoints")
    return response_successful_result(user_standpoints)


@app.route('/get_contract_review_result', methods=["post"])
def get_contract_review_result():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            contract_type_id = in_dict['contract_type_id']
            text = in_dict['contract_content']
            usr = in_dict['user_standpoint_id']
            # print(text)
            # print(contract_type)
            if usr == 'party_a':
                usr = 'Part A'
            elif usr == 'part_b':
                usr = 'Part B'
            else:
                raise Exception("暂时不支持该用户立场")

            acknowledgement = acknowledgement_dict[contract_type_id]

            acknowledgement.review_main(content=text, mode="text")
            res = acknowledgement.review_result
            # 编排返回结果的内容
            result = []
            for review_point, review_result in res.items():
                result.append({
                    "review_point": review_point,
                    "review_result": review_result.get("审核结果", ""),
                    "review_content": review_result.get("内容", ""),
                    "review_content_start": review_result.get("start", -1),
                    "review_content_end": review_result.get("end", -1),
                    "legal_advice": review_result.get("法律建议", ""),
                    "legal_basis": review_result.get("法律依据", ""),
                    "risk_level": review_result.get("风险等级", ""),
                    "risk_point": review_result.get("风险点", "")
                })
            return response_successful_result(result)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)
    except Exception as e:
        logger.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7997, debug=True)  # , use_reloader=False)
