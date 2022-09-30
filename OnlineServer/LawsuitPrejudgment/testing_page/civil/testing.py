#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 28/9/2022 17:14 
@Desc    : None
"""
import requests

URL = "http://172.19.82.198:6080"


def get_anyou_list():
    # return ["借贷纠纷", "劳动社保", "买卖合同", "租赁合同"]
    url = URL + "/reasoning_graph_testing/get_anyou_list"
    resp = requests.get(url)
    resp_json = resp.json()
    return resp_json.get("anyou_list", [])


def get_suqiu_list(anyou):
    # return ["支付劳动劳务报酬", "支付加班工资", "支付双倍工资", "经济补偿金或赔偿金", "劳务受损赔偿", "劳动劳务致损赔偿"]
    url = URL + "/reasoning_graph_testing/get_suqiu_list"
    resp = requests.get(url, {"anyou": anyou})
    resp_json = resp.json()
    return resp_json.get("suqiu_list", [])


def _request(problem, claim_list, fact, question_answers, factor_sentence_list, repeated_question_management):
    url = URL + "/reasoning_graph_testing/get_result"
    body = {
        "problem": problem,
        "claim_list": claim_list,
        "fact": fact,
        "question_answers": question_answers,
        "factor_sentence_list": factor_sentence_list,
        "repeated_question_management": repeated_question_management
    }
    resp = requests.post(url, json=body)
    resp_json = resp.json()
    return resp_json


def get_extracted_features(anyou, suqiu_list, desp):
    # extracted_features = {
    #     "特征": ['无法偿还贷款', '存在借款合同', '不存在借款合同', '出借方未实际提供借款', '借款人逾期未返还借款'],
    #     "对应句子": ['现在公司没有按时还款', '我要求公司按照合同约定', '我要求公司按照合同约定', '返还借款本金130万及利息、违约金、律师费66800元', '现在公司没有按时还款'],
    #     "正向/负向匹配": [1, 1, -1, -1, 1],
    #     "匹配表达式": ['(((没有|没|未|不|非|无|未经|怠于)[^。；，：,;:？！!?\s]*(偿还|归还|偿付|清偿|还款|还清|还本付息|偿清|还债|还账|还钱|付清|结清|返还|支付)))',
    #               '((((签|写|打)[^。；，：,;:？！!?\s]*(借据|借条|欠条|合同|协议)|借据|借条|欠条|合同|协议)))',
    #               '((((签|写|打)[^。；，：,;:？！!?\s]*(借据|借条|欠条|合同|协议)|借据|借条|欠条|合同|协议)))',
    #               '(((给|提供|付|借)[^。；，：,;:？！!?\s]*(款|钱|资金)))',
    #               '(((没有|没|未|不|非|无|未经|怠于)[^。；，：,;:？！!?\s]*(偿还|归还|偿付|清偿|还款|还清|还本付息|偿清|还债|还账|还钱|付清|结清|返还|支付)))']
    # }
    resp = _request(anyou, suqiu_list, desp, {}, {}, None)
    factor_sentence_list = resp.get("factor_sentence_list", [])
    extracted_features = {
        "特征": [],
        "对应句子": [],
        "正向/负向匹配": [],
        "匹配表达式": []
    }

    for sentence, feature, flag, expression in factor_sentence_list:
        extracted_features["特征"].append(feature)
        extracted_features["对应句子"].append(sentence)
        extracted_features["正向/负向匹配"].append(flag)
        extracted_features["匹配表达式"].append(expression)
    return extracted_features


def get_next_question(anyou, suqiu_list, desp, question_answers, factor_sentence_list, repeated_question_management):
    resp = _request(anyou, suqiu_list, desp, question_answers, factor_sentence_list, repeated_question_management)
    next_question_info = resp.get("question_next")
    if next_question_info:
        next_question = str(next_question_info).split(":")[0]
        answers = str(next_question_info).split(":")[1].split(";")
        if resp.get("question_type") == "1":
            single_or_multi = "single"
        else:
            single_or_multi = "multi"
        return True, {"next_question": next_question, "answers": answers, "single_or_multi": single_or_multi,
                      "factor_sentence_list": resp.get("factor_sentence_list"),
                      "repeated_question_management": resp.get("repeated_question_management"),
                      "debug_info": resp.get("debug_info")}
    else:
        return False, {"result": resp.get("result"), "debug_info": resp.get("debug_info")}


if __name__ == '__main__':
    print(get_extracted_features("借贷纠纷", ["还本付息"], "逾期未返还借款"))
