#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 14:12
# @Author  : Adolf
# @Site    : 
# @File    : test_service.py
# @Software: PyCharm
import requests
from Utils import print_run_time

# ip = 'localhost'
ip = '172.19.82.199'
port = 6021


@print_run_time
def test_service(url, data=None):
    r = requests.post(url, json=data)
    result = r.json()
    print("输出:", result)


url1 = "http://%s:%s/getAnyou" % (ip, port)
# test_service(url1)

url2 = "http://%s:%s/getCaseFeature" % (ip, port)
test_service(url2,data={"anyou": "劳动社保_享受失业保险"})

url3 = "http://172.19.82.199:9500/keyword_feature_matching"
request_data = {
    "sentence": "2014年6月，我借给了何三宇、冯群华20000元并写了借条，约定月息3%，在2014年10月14日前一次还清，同时谭学民、蔡金花作了担保人。到期后，何三宇、冯群华迟迟不还款，现在我想让他们按照约定，还我本金及利息。",
    "problem": "借贷纠纷",
    "suqiu": "民间借贷"
}
test_service(url3, request_data)