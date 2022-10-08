#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 14:30 
@Desc    : None
"""
import requests


def get_criminal_result(fact, question_answers, factor_sentence_list):
    body = {
        "fact": fact,
        "question_answers": question_answers,
        "factor_sentence_list": factor_sentence_list
    }
    resp_json = requests.post(url="http://127.0.0.1:8105/get_criminal_result", json=body).json()

    if resp_json.get("success") is False:
        raise Exception("刑事预判接口返还异常: {}".format(resp_json.get("error_msg")))

    next_question_info = resp_json.get("question_next")
    if next_question_info:
        resp_json["next_question"] = str(next_question_info).split(":")[0]
        resp_json["answers"] = str(next_question_info).split(":")[1].split(";")
        resp_json["single_or_multi"] = "single" if resp_json.get("question_type") == "1" else "multi"
        return True, resp_json
    return False, resp_json
