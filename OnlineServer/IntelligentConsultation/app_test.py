#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
from pip import main
import requests

URL = "http://127.0.0.1:8130"


def test_get_query_answer_with_source():
    # 准备测试数据
    body = {
        "question": "公司交不起税怎么办",
        "query_source": "专题"
        # "source_end": "personal"
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/get_query_answer_with_source", json=body).json()
    answer = resp_json.get("answer")

    print(answer)

if __name__ == '__main__':
    test_get_query_answer_with_source()
