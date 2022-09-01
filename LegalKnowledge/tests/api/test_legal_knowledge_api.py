#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 13:18 
@Desc    : None
"""
import time

import requests


LEGAL_KNOWLEDGE_SERVICE_URL = "http://127.0.0.1:8120"


def test_get_columns():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_columns"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")


def test_get_news_by_column_id():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_news_by_column_id"

    for column_id in ["hot_news", "interpret_the_law_by_case", "new_law_express", "study_law_daily"]:
        start_time = time.time()
        resp_json = requests.get(url, {"column_id": column_id, "page_number": 1, "page_size": 10}).json()
        time_cost = time.time() - start_time

        print(resp_json)
        assert resp_json, "column_id:{}".format(column_id)
        assert resp_json.get("success"), "column_id:{}".format(column_id)
        assert resp_json.get("result"), "column_id:{}".format(column_id)
        # assert len(resp_json.get("result")) > 1, "column_id:{}".format(column_id)
        assert time_cost < 0.5, "接口{}用时{}秒，耗时过长。column_id:{}。".format("/get_news_by_column_id", time_cost, column_id)


def test_get_news_by_keyword():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_news_by_keyword"
    start_time = time.time()
    resp_json = requests.get(url, {"keyword": "游泳", "page_number": 1, "page_size": 10}).json()
    time_cost = time.time() - start_time

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    # assert len(resp_json.get("result")) > 1
    assert time_cost < 0.5, "接口{}用时{}秒，耗时过长。".format("/get_news_by_keyword", time_cost)


def test_get_news_by_news_id():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_news_by_news_id"
    start_time = time.time()
    resp_json = requests.get(url, {"news_id": 31}).json()
    time_cost = time.time() - start_time

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    # assert len(resp_json.get("result")) > 1
    assert time_cost < 1, "接口{}用时{}秒，耗时过长。".format("/get_news_by_news_id", time_cost)
