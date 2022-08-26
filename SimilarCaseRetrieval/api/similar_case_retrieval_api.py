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
    return [
        {
            "doc_id": "2b2ed441-4a86-4f7e-a604-0251e597d85e",
            # "doc_id": "24dbed45-904d-4992-aea7-a82000320181",
            "title": "原告王某某与被告郝某某等三人婚约财产纠纷一等婚约财产纠纷一审民事判决书",
            "court": "公主岭市人民法院",
            "judge_date": "2016-04-11",
            "case_number": "（2016）吉0381民初315号",
            "tag": "彩礼 证据 结婚 给付 协议 女方 当事人 登记 离婚",
            "is_guiding_case": True,
            "problem_id": 17
        },
        {
            "doc_id": "ws_c4b1e568-b253-4ac3-afd7-437941f1b17a",
            "title": "原告彭华刚诉被告王金梅、王本忠、田冬英婚约财产纠纷一案",
            "court": "龙山县人民法院",
            "judge_date": "2011-07-12",
            "case_number": "（2011）龙民初字第204号",
            "tag": "彩礼 酒席 结婚 费用 订婚 电视 女方 买家 猪肉",
            "is_guiding_case": False,
            "problem_id": 17
        }
    ]


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
