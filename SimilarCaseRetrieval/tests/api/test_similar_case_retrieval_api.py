#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/10 13:14 
@Desc    : None
"""
import requests

SIMILAR_CASE_RETRIEVAL_URL = "http://101.69.229.138:8140"


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


def test_get_law_document():
    doc_id = "2b2ed441-4a86-4f7e-a604-0251e597d85e"
    resp_json = requests.get(SIMILAR_CASE_RETRIEVAL_URL + "/get_law_document", data={"doc_id": doc_id}).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("result")
