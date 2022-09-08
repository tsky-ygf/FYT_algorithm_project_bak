#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/5 13:24 
@Desc    : None
"""
from flask import Flask
from flask import request
from Utils.http_response import response_successful_result
from IntelligentConsultation.core import intelligent_consultation_service as service
app = Flask(__name__)


@app.route('/get_query_answer', methods=["post"])
def get_query_answer():
    question = request.json.get("question")
    return response_successful_result(service.get_query_answer(question))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8130, debug=False)