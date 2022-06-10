#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 13:12
# @Author  : Adolf
# @Site    : 
# @File    : xz_app.py
# @Software: PyCharm
import traceback
import logging
import logging.handlers
from flask import Flask, request
from flask_cors import CORS
import json

# from pprint import pprint
from paddlenlp import Taskflow

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

schema = ['行政主体', '行为', '尺度', '处罚', '证据', '法律依据', '违法行为']
ie = Taskflow('information_extraction', schema=schema, device_id=1, task_path="model/uie_model/xz2/model_best")


@app.route('/administrative', methods=["post"])
def get_translation_res():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            text = in_dict['content']
            # print(text)
            res = ie(text)
            return json.dumps({'result': res, "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7998, debug=False)  # , use_reloader=False)
