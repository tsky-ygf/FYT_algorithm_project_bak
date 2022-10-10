#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
import requests

URL = "http://127.0.0.1:8105"


def test_get_administrative_type():
    # 执行被测试程序
    resp_json = requests.get(URL + "/get_administrative_type").json()
    print(resp_json)

    # 验证测试条件
    assert resp_json.get("success")
    assert resp_json.get("result")


def test_get_administrative_problem_and_situation_by_type_id():
    # 准备测试数据
    body = {
        "type_id": "tax"
    }

    # 执行被测试程序
    resp_json = requests.get(URL + "/get_administrative_problem_and_situation_by_type_id", params=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json.get("success")
    assert resp_json.get("result")


def test_get_administrative_result():
    # 准备测试数据
    body = {
        "type_id": "tax",
        "situation": "逃避税务机关检查"
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/get_administrative_result", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json.get("success")
    assert resp_json.get("result")
