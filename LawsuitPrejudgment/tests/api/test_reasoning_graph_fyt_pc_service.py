#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/1 11:05 
@Desc    : None
"""
import requests


def test__get_reasoning_graph_result():
    pass


def test_get_civil_report():
    body = {
        "problem": "婚姻继承",
        "claim": "请求分割财产",
        "situation": "财产为婚姻关系存续期间夫妻的共同财产",
        "fact": "周某向本院提交答辩意见：不同意原告的诉讼请求，我不知道原告起诉的数据从哪来的，我只同意付给原告3至5万元。本院认为，离婚后，一方以尚有夫妻共同财产未处理为由向人民法院起诉请求分割的，经审查该财产确属离婚时未涉及的夫妻共同财产，人民法院应当依法予以分割。本案中，原告提出有211608元的夫妻共同财产在离婚时没有分割，现主张一半的权利。但对于其中离婚前被告取出的47800元，原告虽然提交被告工资收入等明细清单，但原告在共同生活期间理应知被告收入情况及消费情况，且原告并无证据证明被告婚前取出的47800元未用于生活消费，故本院认定离婚时尚未处理的163808.2元为共同财产。对于未认定的47800元，如果原告有证据证明未用于生活消费，仍可继续主张。本院审理中，被告提出只同意给付原告3至5万元，因没有事实及法律依据，本院不予支持。综上，本院对原告诉讼请求的合理部分，予以支持。"
    }

    resp_json = requests.post("http://172.19.82.199:5088/get_civil_report", json=body).json()
    print(resp_json)

    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("result")
