#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  :
@Time    : 2022/8/17 14:14 
@Desc    : None
"""
import json

import requests


def get_filter_conditions():
    url = "http://0.0.0.0:8162/get_filter_conditions_of_law"
    resp_json = requests.get(url).json()
    print(type(resp_json))
    print(resp_json)
    print(json.loads(resp_json)["result"])
    # assert resp_json
    # assert resp_json.get("success")
    # assert resp_json.get("result")
    # assert resp_json["result"].get("types_of_law")
    # assert resp_json["result"].get("timeliness")
    # assert resp_json["result"].get("scope_of_use")
    pass


def search_laws():
    url = "http://101.69.229.138:8135/search_laws"
    url = "http://0.0.0.0:8162/search_laws"
    body = {
        "query": "侵权",
        "filter_conditions": {
            "types_of_law": ["地方性法规"],
            "timeliness": ["全部"],
            "scope_of_use": ["广东省"],
        },
        "page_number": 1,
        "page_size": 10,
    }

    resp_json = requests.post(url, json=body).json()
    print(resp_json)

    assert resp_json
    # assert resp_json.get("success")
    # assert resp_json.get("result")
    # assert resp_json.get("total_amount")
    # assert "law_name" in resp_json["result"][0]


def get_law_by_law_id():
    url = "http://172.19.82.199:8160/get_law_by_law_id"
    param = {
        "law_id": "flfg_result_falv"
        + "_SEP_"
        + "5a43120b27fe0457634a7420283b4aad"
    }
    resp_json = requests.get(url, params=param).json()
    print(resp_json)

    assert resp_json.get("success")
    assert resp_json.get("result")
    assert resp_json["result"]["law_id"] == param["law_id"]


if __name__ == "__main__":
    get_filter_conditions()
    search_laws()
    # get_law_by_law_id()
