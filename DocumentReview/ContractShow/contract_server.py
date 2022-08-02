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
            contract_type = in_dict['contract_type']
            text = in_dict['contract_content']
            usr = in_dict['user_standpoint']
            # print(text)
            print(contract_type)
            if usr == 'party_a':
                usr = 'Part A'
            else:
                usr = 'Part B'

            if contract_type == "借条":
                config_path = "DocumentReview/Config/jietiao.csv"
                model_path = "model/uie_model/model_best/"
            elif contract_type == "借款合同":
                config_path = "DocumentReview/Config/jiekuan.csv"
                model_path = "model/uie_model/jkht/model_best/"
            elif contract_type == "劳动合同":
                config_path = "DocumentReview/Config/laodong.csv"
                model_path = "model/uie_model/laodong/model_best/"
            elif contract_type == "租房合同":
                config_path = "DocumentReview/Config/fangwuzulin.csv"
                model_path = "model/uie_model/fwzl/model_best/"
            elif contract_type == "买卖合同":
                config_path = "DocumentReview/Config/maimai.csv"
                model_path = "model/uie_model/maimai/model_best/"
            elif contract_type == "劳务合同":
                config_path = "DocumentReview/Config/laowu.csv"
                model_path = 'model/uie_model/guyong/model_best/'
            else:
                raise Exception("暂时不支持该合同类型")

            acknowledgement = BasicUIEAcknowledgement(config_path=config_path,
                                                      model_path=model_path,
                                                      usr=usr,
                                                      device_id=-1)

            acknowledgement.review_main(content=text, mode="text")
            res = acknowledgement.review_result
            return response_successful_result(res)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logger.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7997, debug=True)  # , use_reloader=False)
