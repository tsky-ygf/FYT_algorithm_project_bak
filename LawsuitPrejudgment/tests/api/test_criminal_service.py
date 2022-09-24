#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2/9/2022 16:35 
@Desc    : None
"""
import time

import requests


def test_should_ask_question_when_get_criminal_result():
    url = "http://127.0.0.1:5080/get_criminal_result"
    data = {
        "fact": "2022年8月12日，罗某某利用螺丝刀撬开房间门锁进入某市某区某栋某单元某层某房间内，窃得现金50000元。2022年8月12日，趁邻居卢某家无人在家，从卢某家厨房后窗翻进其家，盗走现金50000元。",
        "question_answers": {},
        "factor_sentence_list": []
    }
    resp_json = requests.post(url, json=data).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") == "盗窃人年龄未满十六周岁或者精神状态不正常？:是;否"
    assert resp_json.get("question_type") == "1"
    assert resp_json.get("result") is None


def test_should_get_report_when_get_criminal_result():
    url = "http://127.0.0.1:5080/get_criminal_result"
    data = {
        "fact": "我偷钱了。",
        "question_answers": {
            "盗窃人年龄未满十六周岁或者精神状态不正常？:是;否": "是"
        },
        "factor_sentence_list": []
    }
    start_time = time.time()
    resp_json = requests.post(url, json=data).json()
    time_cost = time.time() - start_time

    print(resp_json)
    print("time cost:{}".format(time_cost))

    assert resp_json
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") is None
    assert resp_json.get("question_type") == "1"
    assert resp_json.get("support") is True
    assert resp_json.get("result")
    assert time_cost < 5


def test_should_get_unsupport_report_when_get_criminal_result():
    url = "http://127.0.0.1:5080/get_criminal_result"
    data = {
        "fact": "罗某贩毒被抓。",
        "question_answers": {},
        "factor_sentence_list": []
    }
    resp_json = requests.post(url, json=data).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") is None
    assert resp_json.get("question_type") == "1"
    assert resp_json.get("support") is False
    assert "unsupport_reason" in resp_json.get("result")
