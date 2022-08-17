#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/17 11:53 
@Desc    : 法条检索模块的接口
"""

from flask import Flask
from flask import request

from Utils.io import read_json_attribute_value
from Utils.http_response import response_successful_result


app = Flask(__name__)


@app.route('/get_filter_conditions', methods=["get"])
def get_columns():
    filer_conditions = read_json_attribute_value("RelevantLaws/api/filter_conditions.json", "filter_conditions")
    return response_successful_result(filer_conditions)


# @app.route('/search_laws', methods=["post"])
# def get_columns():
#     query = request.json.get("query")
#     return response_successful_result(service.get_columns())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8120, debug=True)