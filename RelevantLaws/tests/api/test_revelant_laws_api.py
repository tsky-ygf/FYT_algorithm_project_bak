#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/17 14:14 
@Desc    : None
"""
import requests


def test_get_filter_conditions():
    url = "http://101.69.229.138:8135/get_filter_conditions"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert resp_json["result"].get("types_of_law")
    assert resp_json["result"].get("timeliness")
    pass


def test_search_laws():
    url = "http://101.69.229.138:8135/search_laws"
    body = {
        "query": "侵权",
        "filter_conditions": {
            "types_of_law": ["法律", "司法解释"],
            "timeliness": ["有效", "已修改"],
            "size": 10
        }
    }

    resp_json = requests.post(url, json=body).json()
    print(resp_json)

    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert "law_name" in resp_json["result"][0]
