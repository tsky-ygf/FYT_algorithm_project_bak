#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 20/10/2022 9:14 
@Desc    : None
"""
import requests

URL = "http://127.0.0.1:8133"


def test_criminal_judgment1():
    # 准备测试数据
    body = {
        "dialogue_history": {
            "user_input": "我偷了舍友500块。",
            "question_answers": None
        },
        "dialogue_state": {
            "domain": "criminal",
            "problem": None,
            "claim_list": None,
            "other": None
        }
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    # print(resp_json)

    # 验证测试条件
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("next_action")
    assert resp_json["next_action"]["action_type"] == "ask"
    assert resp_json["next_action"]["content"]["question"] == "盗窃人年龄未满十六周岁或者精神状态不正常？"
    assert "是" in resp_json["next_action"]["content"]["candidate_answers"]

    # 准备测试数据
    context = resp_json["dialogue_state"]["other"]
    body = {
        "dialogue_history": {
            "user_input": "我偷了舍友500块。",
            "question_answers": [
                {
                    "question": "盗窃人年龄未满十六周岁或者精神状态不正常？",
                    "candidate_answers": ["是","否"],
                    "question_type": "single",
                    "other": {
                        "circumstances": "前提"
                    },
                    "user_answer": ["是"]
                }
            ]
        },
        "dialogue_state": {
            "domain": "criminal",
            "problem": "盗窃",
            "claim_list": ["量刑推荐"],
            "other": context
        }
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("next_action")
    assert resp_json["next_action"]["action_type"] == "report"
    assert "report" in resp_json["next_action"]["content"]


def test_criminal_judgment2():
    # 准备测试数据
    body = {
        "dialogue_history": {
            "user_input": "1111111",
            "question_answers": None
        },
        "dialogue_state": {
            "domain": "criminal",
            "problem": None,
            "claim_list": None,
            "other": None
        }
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("next_action")
    assert resp_json["next_action"]["action_type"] == "report"
    assert "report" in resp_json["next_action"]["content"]
