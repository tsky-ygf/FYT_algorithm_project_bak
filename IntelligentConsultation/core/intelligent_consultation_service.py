#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/5 13:27 
@Desc    : None
"""
import requests


def get_query_answer(question: str):
    url = "http://172.19.82.198:5050/get_query_answer"
    resp_json = requests.post(url, json={"question":question}).json()
    return {
        "answer": resp_json.get("answer")
    }
