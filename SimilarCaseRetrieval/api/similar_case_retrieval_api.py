#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/10 13:14 
@Desc    : None
"""
from flask import Flask, request
from Utils.http_response import response_successful_result
from SimilarCaseRetrieval.core import similar_case_retrieval_service as service

app = Flask(__name__)


def _preprocess(request_json):
    problem = request_json.get("problem", "")
    if problem is None or str(problem).strip() == "":
        problem = ""

    claim_list = request_json.get("claim_list", [])
    if claim_list is None or len(claim_list) == 0:
        claim_list = []

    fact = request_json.get("fact")
    return problem, claim_list, fact


@app.route("/get_similar_cases", methods=["post"])
def get_similar_cases():
    problem, claim_list, fact = _preprocess(request.json)
    return response_successful_result(service.get_similar_cases(problem, claim_list, fact))


@app.route('/get_law_document', methods=["get"])
def get_law_document():
    doc_id = request.args.get("doc_id")
    return response_successful_result(service.get_law_document(doc_id))
