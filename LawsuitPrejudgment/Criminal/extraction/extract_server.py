#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/25 17:37
# @Author  : Adolf
# @Site    : 
# @File    : extract_server.py
# @Software: PyCharm
import traceback
from flask import Flask, request
from flask_cors import CORS
from loguru import logger

import json

from LawsuitPrejudgment.Criminal.extraction.feature_extraction import init_extract, post_process_uie_results
from Utils.http_response import response_successful_result, response_failed_result

criminal_list = ['theft', 'provide_drug']
predictor_dict = {}
for criminal_type in criminal_list:
    model_path = "model/uie_model/export_cpu/{}/inference".format(criminal_type)
    predictor_dict[criminal_type] = init_extract(criminal_type=criminal_type)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/information_result', methods=["post"])
def get_information_result():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            _criminal_type = in_dict['criminal_type']
            _fact = in_dict['fact']
            # result = predictor_dict[_criminal_type].predict([_fact])
            result = post_process_uie_results(predictor_dict[_criminal_type], _criminal_type, _fact)
            return response_successful_result(result)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)
    except Exception as e:
        logger.info(traceback.format_exc())
        return response_failed_result(traceback.format_exc())


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7777, debug=False)  # , use_reloader=False)
