#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 17:45 
@Desc    : None
"""
import requests


def get_criminal_result(fact, question_answers, factor_sentence_list):
    body = {
        "fact": fact,
        "question_answers": question_answers,
        "factor_sentence_list": factor_sentence_list
    }
    return requests.post(url="http://127.0.0.1:8100/get_criminal_result", json=body)