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
