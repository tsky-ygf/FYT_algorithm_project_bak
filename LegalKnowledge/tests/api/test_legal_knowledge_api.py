#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 13:18 
@Desc    : None
"""
import requests


LEGAL_KNOWLEDGE_SERVICE_URL = "http://101.69.229.138:8120"


def test_get_columns():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_columns"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")


def test_get_news_by_column_id():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_news_by_column_id"
    resp_json = requests.get(url, {"column_id": "hot_news"}).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert len(resp_json.get("result")) > 1


def test_get_news_by_keyword():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_news_by_keyword"
    resp_json = requests.get(url, {"keyword": "游泳"}).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
    assert len(resp_json.get("result")) > 1