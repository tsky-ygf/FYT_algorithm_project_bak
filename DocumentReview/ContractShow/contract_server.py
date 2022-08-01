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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/contractreview', methods=["post"])
def get_translation_res():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            contract_type = in_dict['contract_type']
            text = in_dict['content']
            usr = in_dict['usr']
            # print(text)
            print(contract_type)
            if usr == '甲方':
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
                config_path = "DocumentReview/Config/labor.csv"
                model_path = "model/uie_model/labor/model_best/"
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
            return json.dumps({'result': res, "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logger.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7997, debug=True)  # , use_reloader=False)
