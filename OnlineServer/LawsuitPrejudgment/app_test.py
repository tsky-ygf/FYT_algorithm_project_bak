#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
import requests

URL = "http://127.0.0.1:8133"


# def test_get_administrative_type():
#     # 执行被测试程序
#     resp_json = requests.get(URL + "/get_administrative_type").json()
#     print(resp_json)
#
#     # 验证测试条件
#     assert resp_json.get("success")
#     assert resp_json.get("result")


# def test_get_administrative_problem_and_situation_by_type_id():
#     # 准备测试数据
#     body = {
#         "type_id": "tax"
#     }
#
#     # 执行被测试程序
#     resp_json = requests.get(URL + "/get_administrative_problem_and_situation_by_type_id", params=body).json()
#     print(resp_json)
#
#     # 验证测试条件
#     assert resp_json.get("success")
#     assert resp_json.get("result")


# def test_get_administrative_result():
#     # 准备测试数据
#     body = {
#         "type_id": "tax",
#         "situation": "逃避税务机关检查"
#     }
#
#     # 执行被测试程序
#     resp_json = requests.post(URL + "/get_administrative_result", json=body).json()
#     print(resp_json)
#
#     # 验证测试条件
#     assert resp_json.get("success")
#     assert resp_json.get("result")


def test_get_civil_problem_summary():
    # 执行被测试程序
    resp_json = requests.get(URL + "/get_civil_problem_summary").json()
    print(resp_json)

    # 验证测试条件
    assert resp_json.get("success")
    assert resp_json.get("value")


def test_get_template_by_problem_id():
    # 准备测试数据
    body = {
        "problem_id": 1564
    }

    # 执行被测试程序
    resp_json = requests.get(URL + "/get_template_by_problem_id", params=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json.get("success")
    assert resp_json.get("value")
    assert resp_json["value"]["template"]


def test_get_claim_list_by_problem_id():
    # 准备测试数据
    body = {
        "problem_id": 1536,
        "fact": "婚后男的方父母出资首得到付，夫妻名义贷款还贷，房产证只写男方名，离婚后财产如何分配"
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/get_claim_list_by_problem_id", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json.get("success")
    assert resp_json.get("value")
    assert next((item.get("claim") for item in resp_json["value"] if item.get("claim") == "房产分割"), None)


def test_should_ask_next_question_when_reasoning_graph_result():
    # 准备测试数据
    body = {
        "problem": "婚姻家庭",
        "claim_list": ["请求离婚"],
        "fact": "男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。",
        "question_answers": {},
        "factor_sentence_list": []
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/reasoning_graph_result", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("question_next")
    assert resp_json.get("result") is None


def test_show_have_report_when_reasoning_graph_result():
    # 准备测试数据
    body = {
        "problem": "婚姻家庭",
        "claim_list": ["返还彩礼", "房产分割"],
        "fact": "111111111111111111111111111111",
        "question_answers": {
            "是否存在以下情形？:双方未共同生活;给付彩礼导致给付方生活困难;双方未登记结婚;以上都没有": "双方未共同生活",
            "双方是否办理结婚登记？:是;否": "否",
            "是否存在以下情形？:非商品房为婚后所得;非商品房为婚前所得;婚前一方全款买房;婚前一方分期买房;婚前一方父母买房;婚前双方父母共同出资买房;婚前双方共同分期买房;婚后一方以个人财产买房;婚后双方共同出资买房;婚后一方父母全款买房;婚后双方父母出资买房;房产登记在自己名下;房产登记在双方名下;房产登记在一方名下;以上都没有": "非商品房为婚前所得"
        },
        "factor_sentence_list": []
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/reasoning_graph_result", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("question_next") is None
    assert resp_json.get("result")


def test_administrative_judgment1():
    # 准备测试数据
    body = {
        "dialogue_history": {
            "user_input": None,
            "question_answers": None
        },
        "dialogue_state": {
            "domain": "administrative",
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
    assert resp_json["next_action"]["action_type"] == "ask"
    assert resp_json["next_action"]["content"]["question"] == "请选择你遇到的纠纷类型"
    assert "税务处罚预判" in resp_json["next_action"]["content"]["candidate_answers"]


def test_administrative_judgment2():
    # 准备测试数据
    body = {
        "dialogue_history": {
            "user_input": None,
            "question_answers": [
                {
                    "question": "请选择你遇到的纠纷类型",
                    "candidate_answers": ["税务处罚预判","治安处罚预判","交警处罚预判"],
                    "question_type": "single",
                    "other": {
                        "slot": "anyou"
                    },
                    "user_answer": "税务处罚预判"
                }
            ]
        },
        "dialogue_state": {
            "domain": "administrative",
            "problem": "税务处罚预判",
            "claim_list": ["行政预判"],
            "other": {
                "slots": {
                    "anyou": None,
                    "problem": None,
                    "situation": None
                }
            }
        }
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("next_action")
    assert resp_json["next_action"]["action_type"] == "ask"
    assert resp_json["next_action"]["content"]["question"] == "请选择你遇到的问题"
    assert "发票问题" in resp_json["next_action"]["content"]["candidate_answers"]


def test_administrative_judgment3():
    # 准备测试数据
    body = {
        "dialogue_history": {
            "user_input": None,
            "question_answers": [
                {
                    "question": "请选择你遇到的纠纷类型",
                    "candidate_answers": ["税务处罚预判","治安处罚预判","交警处罚预判"],
                    "question_type": "single",
                    "other": {
                        "slot": "anyou"
                    },
                    "user_answer": "税务处罚预判"
                },
                {
                    "question": "请选择你遇到的问题",
                    "candidate_answers": ["发票问题", "账簿、记账问题", "申报问题", "缴税问题"],
                    "question_type": "single",
                    "other": {
                        "slot": "problem"
                    },
                    "user_answer": "发票问题"
                }
            ]
        },
        "dialogue_state": {
            "domain": "administrative",
            "problem": "税务处罚预判",
            "claim_list": ["行政预判"],
            "other": {
                "slots": {
                    "anyou": "税务处罚预判",
                    "problem": None,
                    "situation": None
                }
            }
        }
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    print(resp_json)

    # 验证测试条件
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("next_action")
    assert resp_json["next_action"]["action_type"] == "ask"
    assert resp_json["next_action"]["content"]["question"] == "请选择具体的情形"
    assert "虚开增值税发票" in resp_json["next_action"]["content"]["candidate_answers"]


def test_administrative_judgment4():
    # 准备测试数据
    body = {
        "dialogue_history": {
            "user_input": None,
            "question_answers": [
                {
                    "question": "请选择你遇到的纠纷类型",
                    "candidate_answers": ["税务处罚预判","治安处罚预判","交警处罚预判"],
                    "question_type": "single",
                    "other": {
                        "slot": "anyou"
                    },
                    "user_answer": "税务处罚预判"
                },
                {
                    "question": "请选择你遇到的问题",
                    "candidate_answers": ["发票问题", "账簿、记账问题", "申报问题", "缴税问题"],
                    "question_type": "single",
                    "other": {
                        "slot": "problem"
                    },
                    "user_answer": "发票问题"
                },
                {
                    "question": "请选择具体的情形：",
                    "candidate_answers": ["没有真实的业务、资金往来", "虚开增值税发票"],
                    "question_type": "single",
                    "other": {
                        "slot": "situation"
                    },
                    "user_answer": "虚开增值税发票"
                }
            ]
        },
        "dialogue_state": {
            "domain": "administrative",
            "problem": "税务处罚预判",
            "claim_list": ["行政预判"],
            "other": {
                "slots": {
                    "anyou": "税务处罚预判",
                    "problem": "发票问题",
                    "situation": None
                }
            }
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
