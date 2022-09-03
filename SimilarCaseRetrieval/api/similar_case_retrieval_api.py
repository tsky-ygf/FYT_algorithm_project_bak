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
from SimilarCaseRetrieval.core.relevant_cases_search import get_case_search_result
from typing import List

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


@app.route('/get_filter_conditions_of_case', methods=["get"])
def get_filter_conditions_of_case():
    filter_conditions = {
        "type_of_case": {
            "name": "案件类型",
            "is_multiple_choice": True,
            "value": [
                "合同纠纷",
                "婚姻家庭",
                "刑事案件"
            ]
        },
        "court_level": {
            "name": "法院层级",
            "is_multiple_choice": True,
            "value": [
                "最高",
                "高级",
                "中级",
                "基层"
            ]
        },
        "type_of_document": {
            "name": "文书类型",
            "is_multiple_choice": True,
            "value": [
                "判决",
                "裁定",
                "调解"
            ]
        },
        "region": {
            "name": "地域",
            "is_multiple_choice": True,
            "value": [
                "江苏",
                "浙江",
                "福建",
                "山东"
            ]
        }
    }
    return response_successful_result(filter_conditions)


def _get_search_result(query, filter_conditions):
    search_result = get_case_search_result(query,
                                           filter_conditions.get("type_of_case"),
                                           filter_conditions.get("court_level"),
                                           filter_conditions.get("type_of_document"),
                                           filter_conditions.get("region"),
                                           filter_conditions.get("size", 10))
    return _construct_result_format(search_result)


@app.route('/search_cases', methods=["post"])
def search_cases():
    query = request.json.get("query")
    filter_conditions = request.json.get("filter_conditions")
    result = _get_search_result(query, filter_conditions)
    # TODO: 实现用于分页的total_amount
    return response_successful_result(result, {"total_amount": len(result)})


@app.route('/get_law_document', methods=["get"])
def get_law_document():
    doc_id = request.args.get("doc_id")
    result = service.get_criminal_law_document(doc_id)
    if result:
        response_successful_result(result)

    mock_doc_id = "24dbed45-904d-4992-aea7-a82000320181"
    return response_successful_result(service.get_criminal_law_document(mock_doc_id))


def _construct_result_format(search_result) -> List:
    result = []
    for index, row in search_result.iterrows():

        result.append({
            "doc_id": row['uq_id'],
            "court": row['faYuan_name'],
            "case_number": row['event_num']})
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8801, debug=True)