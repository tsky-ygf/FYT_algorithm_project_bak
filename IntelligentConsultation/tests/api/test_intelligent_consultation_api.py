#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/5 13:27 
@Desc    : None
"""
import requests

INTELLIGENT_CONSULTATION_API_URL = "http://101.69.229.138:7130"


def test_get_query_answer():
    url = INTELLIGENT_CONSULTATION_API_URL + "/get_query_answer"
    query = "为他人或自己开具与实际经营业务情况不符的发票会有什么后果？"

    resp_json = requests.post(url, json={"question": query}).json()

    print(resp_json)
    assert resp_json
    assert resp_json["success"]
    pass
