#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 13:17 
@Desc    : 普法常识模块的接口
"""
import traceback
from loguru import logger
import requests
from flask import Flask
from flask import request
from Utils.http_response import response_successful_result, response_failed_result
from LegalKnowledge.core import legal_knowledge_service as service


app = Flask(__name__)


@app.route('/get_columns', methods=["get"])
def get_columns():
    return response_successful_result(service.get_columns())


@app.route('/get_news_by_column_id', methods=["get"])
def get_news_by_column_id():
    column_id = request.args.get("column_id")
    if column_id:
        return response_successful_result(service.get_news_by_column_id(column_id))
    return response_failed_result("No parameter: column_id")


@app.route('/get_news_by_keyword', methods=["post"])
def get_news_by_keyword():
    keyword = request.json.get("keyword")
    if keyword:
        return response_successful_result(service.get_news_by_keyword(keyword))
    return response_failed_result("No parameter: keyword")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8120, debug=True)
