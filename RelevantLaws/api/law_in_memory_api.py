#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 4/9/2022 20:14 
@Desc    : None
"""
from flask import Flask
from flask import request

from Utils.http_response import response_successful_result

app = Flask(__name__)

_memory = dict()


@app.route('/get_law_in_memory', methods=["post"])
def get_law_in_memory():
    law_id = request.json.get("law_id")
    return response_successful_result(_memory.get(law_id))


@app.route('/store_law_in_memroy', methods=["post"])
def store_law_in_memroy():
    law_id = request.json.get("law_id")
    _memory[law_id] = {
        "law_id": law_id,
        "law_name": request.json.get("law_name"),
        "law_item": request.json.get("law_item"),
        "law_content": request.json.get("law_content")
    }
    return response_successful_result(None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5090, debug=False)