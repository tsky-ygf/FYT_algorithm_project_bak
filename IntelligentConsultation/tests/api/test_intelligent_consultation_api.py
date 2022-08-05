#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/5 13:27 
@Desc    : None
"""
import requests

INTELLIGENT_CONSULTATION_API_URL = "http://101.69.229.138:8130"


def test_get_query_answer():
    url = INTELLIGENT_CONSULTATION_API_URL + "/get_query_answer"
    query = "夫妻婚内，男方私自在外以夫妻名义借钱，算夫妻共同债务吗？"

    resp_json = requests.post(url, json={"question": query}).json()

    print(resp_json)
    assert resp_json
    assert resp_json["success"]
    pass
