#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 14:12
# @Author  : Adolf
# @Site    : 
# @File    : test_service.py
# @Software: PyCharm
import requests
from pprint import pprint
from Utils import print_run_time

# ip = '172.19.82.199'
ip = "localhost"
port = 6021


@print_run_time
def test_service(url, data=None):
    r = requests.post(url, json=data)
    # print(f"response:{r}")
    result = r.json()
    pprint(result)


url1 = "http://%s:%s/getAnyou" % (ip, port)
# test_service(url1)

url2 = "http://%s:%s/getCaseFeature" % (ip, port)

test_service(url2, data={"anyou": "劳动社保_享受失业保险"})
