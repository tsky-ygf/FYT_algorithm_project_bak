#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/17 14:14 
@Desc    : None
"""
import requests

from ProfessionalSearch.RelevantLaws.api.constants import SEPERATOR_BETWEEN_LAW_TABLE_AND_ID


def test_get_filter_conditions():
    url = "http://101.69.229.138:8135/get_filter_conditions_of_law"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert resp_json["result"].get("types_of_law")
    assert resp_json["result"].get("timeliness")
    assert resp_json["result"].get("scope_of_use")
    pass


def test_search_laws():
    url = "http://101.69.229.138:8135/search_laws"
    body = {
        "query": "侵权",
        "filter_conditions": {
            "types_of_law": [
                "法律",
                "司法解释"
            ],
            "timeliness": [
                "有效",
                "已修改"
            ],
            "scope_of_use": [
                "全国"
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
    assert "law_name" in resp_json["result"][0]


def test_get_law_by_law_id():
    url = "http://101.69.229.138:8135/get_law_by_law_id"
    param = {
        "law_id": "flfg_result_falv" + SEPERATOR_BETWEEN_LAW_TABLE_AND_ID + "5a43120b27fe0457634a7420283b4aad"
    }
    resp_json = requests.get(url, params=param).json()
    print(resp_json)

    assert resp_json.get("success")
    assert resp_json.get("result")
    assert resp_json["result"]["law_id"] == param["law_id"]
