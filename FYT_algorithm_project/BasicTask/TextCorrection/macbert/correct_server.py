#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 17:29
# @Author  : Adolf
# @Site    : 
# @File    : correct_server.py
# @Software: PyCharm
import traceback
from flask import Flask
import json
from flask import request
from loguru import logger

from BasicTask.TextCorrection.macbert.predict import MacbertCorrected

app = Flask(__name__)
m = MacbertCorrected()


@app.route('/macbert_correct', methods=["get", "post"])
def get_macbert_correct_res():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            text = in_dict['text']
            res = m(text)

            return json.dumps({
                "success": True,
                "error_msg": "",
                "result": res,
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error_msg": "request data is none."
            }, ensure_ascii=False)
    except Exception as e:
        logger.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6598, debug=True)
