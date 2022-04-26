#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 14:05
# @Author  : Adolf
# @Site    : 
# @File    : app_service.py
# @Software: PyCharm
import json
import traceback
import logging
import logging.handlers
from flask import Flask, request, jsonify
from flask_cors import CORS
from DataCentric.AnnotationTool.CaseFeatureTool.entity_annotation import *

"""
推理图谱的接口
"""
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/getAnyou', methods=["post"])
def get_anyou():
    try:
        anyou_list = get_anyou_list()
        return json.dumps(
            {'AnyouList': anyou_list, "error_msg": "", "status": 0},
            ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


@app.route('/getCaseFeature', methods=["post"])
def get_case_feature():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            anyou = in_dict['anyou']
            anyou_list = get_case_feature_dict(anyou_name=anyou)
            return json.dumps(
                {'AnyouList': anyou_list, "error_msg": "", "status": 0},
                ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)

@app.route('/getBaseData', methods=["post"])
def get_base_data():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            anyou = in_dict['anyou']
            # anyou_list = get_case_feature_dict(anyou_name=anyou)
            return json.dumps(
                {"case_id": "", "本院认为":"","原告诉称":"","本院查明":"","error_msg": "", "status": 0},ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6021, debug=True)  # , use_reloader=False)
