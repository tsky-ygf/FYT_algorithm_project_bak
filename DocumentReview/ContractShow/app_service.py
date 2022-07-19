#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 13:07
# @Author  : Adolf
# @Site    : 
# @File    : app_service.py
# @Software: PyCharm
import traceback
import logging
import logging.handlers
from flask import Flask, request
from flask_cors import CORS
import json

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
            # print(text)
            print(contract_type)
            if contract_type == '借条':
                from DocumentReview.ContractReview_bak.loan_review import LoanUIEAcknowledgement

                acknowledgement = LoanUIEAcknowledgement(
                    config_path="DocumentReview/Config_bak/LoanConfig/jietiao_20220531.csv",
                    log_level="info",
                    model_path="model/uie_model/model_best/")
            elif contract_type == '借款合同':
                from DocumentReview.ContractReview_bak.loan_contract_review import LoanContractUIEAcknowledgement

                acknowledgement = LoanContractUIEAcknowledgement(
                    config_path="DocumentReview/Config_bak/LoanConfig/jiekuan_20220605.csv",
                    log_level="info",
                    model_path="model/uie_model/jkht/model_best")
            elif contract_type == '劳动合同':
                from DocumentReview.ContractReview_bak.labor_review import LaborUIEAcknowledgement

                acknowledgement = LaborUIEAcknowledgement(
                    config_path="DocumentReview/Config_bak/LaborConfig/labor_20220615.csv",
                    log_level="info",
                    model_path="model/uie_model/labor/model_best")
            else:
                raise Exception("暂时不支持该合同类型")

            acknowledgement.review_main(content=text, mode="text")
            res = acknowledgement.review_result
            return json.dumps({'result': res, "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7997, debug=True)  # , use_reloader=False)
