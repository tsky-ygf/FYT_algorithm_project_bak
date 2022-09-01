#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/10 13:14 
@Desc    : None
"""
import requests

SIMILAR_CASE_RETRIEVAL_URL = "http://127.0.0.1:8140"


def _get_similar_cases(req_data):
    return requests.post(SIMILAR_CASE_RETRIEVAL_URL + "/get_similar_cases", json=req_data).json()


def test_get_similar_cases_with_problem_and_claim():
    problem = "婚姻继承"
    claim_list = ["财产分割"]
    fact = "男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。"

    resp_json = _get_similar_cases({"problem": problem, "claim_list": claim_list, "fact": fact})

    print(resp_json)
    assert resp_json
    assert resp_json.get("result")


def test_get_similar_cases_without_problem_and_claim():
    fact = "请我和老婆于2017年11月结婚，结婚时给女方彩礼30万，现要离婚，彩礼可以要回来么"

    resp_json = _get_similar_cases({"fact": fact})

    print(resp_json)
    assert resp_json
    assert resp_json.get("result")


def test_get_filter_conditions():
    url = "http://127.0.0.1:8140/get_filter_conditions_of_case"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert resp_json["result"].get("type_of_case")
    assert resp_json["result"].get("court_level")
    assert resp_json["result"].get("type_of_document")
    assert resp_json["result"].get("region")
    pass


def test_search_cases():
    url = "http://127.0.0.1:8140/search_cases"
    body = {
        "query": "离婚",
        "filter_conditions": {
            "type_of_case": [
                "合同纠纷",
                "婚姻家庭",
                "刑事案件"
            ],
            "court_level": [
                "最高",
                "高级",
                "中级",
                "基层"
            ],
            "type_of_document": [
                "判决",
                "裁定",
                "调解"
            ],
            "region": [
                "江苏",
                "浙江",
                "福建",
                "山东"
            ]
        },
        "page_number": 1,
        "page_size": 10
    }

    resp_json = requests.post(url, json=body).json()
    print(resp_json)

    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert resp_json.get("total_amount")
    assert "doc_id" in resp_json["result"][0]


def test_get_law_document():
    doc_id = "2b2ed441-4a86-4f7e-a604-0251e597d85e"
    resp_json = requests.get(SIMILAR_CASE_RETRIEVAL_URL + "/get_law_document", data={"doc_id": doc_id}).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("result")


def test_get_criminal_law_document():
    doc_id = "24dbed45-904d-4992-aea7-a82000320181"
    url = "http://127.0.0.1:8140/get_law_document"
    resp_json = requests.get(url, params={"doc_id": doc_id}).json()

    print(resp_json)
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert resp_json["result"]["doc_id"] == doc_id
    assert resp_json["result"]["doc_title"] == "李某汉、黄某娟走私、贩卖、运输、制造毒品一审刑事判决书"
