#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 18:54
# @Author  : Adolf
# @Site    : 
# @File    : time_bingfa.py
# @Software: PyCharm
import time
import gevent
import requests

LEGAL_KNOWLEDGE_SERVICE_URL = "http://101.69.229.138:8120"


def get_content():
    url = LEGAL_KNOWLEDGE_SERVICE_URL + "/get_news_by_column_id"

    for column_id in ["hot_news", "interpret_the_law_by_case", "new_law_express", "study_law_daily"]:
        resp_json = requests.get(url, {"column_id": column_id, "page_number": 1, "page_size": 10}).json()


def call_gevent(count):
    """调用gevent 模拟高并发"""
    begin_time = time.time()
    run_gevent_list = []
    for i in range(count):
        print('--------------%d--Test-------------' % i)
        run_gevent_list.append(gevent.spawn(get_content))
    gevent.joinall(run_gevent_list)
    end = time.time()
    print('单次测试时间（平均）s:', (end - begin_time) / count)
    print('累计测试时间 s:', end - begin_time)


# test_count = 10
# call_gevent(count=test_count)


def get_recommend_news():
    url = "http://47.111.0.124:8110/recommend_laws"
    res = requests.post(url=url).text
    print(res)
    return res

get_recommend_news()