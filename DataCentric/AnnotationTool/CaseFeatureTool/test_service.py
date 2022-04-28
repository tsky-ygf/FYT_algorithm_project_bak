#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 14:12
# @Author  : Adolf
# @Site    : 
# @File    : test_service.py
# @Software: PyCharm
import requests
from Utils import print_run_time

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
# test_service(url2,data={"anyou": "劳动社保_享受失业保险"})

sentence = "本院认为，借据是出借人据以证明其与借款人之间借贷关系成立及交付借款的直接证据，原告朱楠琦与被告钟赛赛之间的民间借贷关系依法成立生效。" \
           "原告自认被告按约支付利息至2018年7月6日，被告未予以抗辩，本院予以采信。现原告要求被告归还"

url3 = "http://172.19.82.199:9500/keyword_feature_matching"
request_data = {
    "sentence": sentence,
    "problem": "借贷纠纷",
    "suqiu": None
}
# test_service(url3, request_data)

url4 = "http://%s:%s/getBaseData" % (ip, port)
# test_service(url4,data={"anyou": "借贷纠纷_民间借贷"})

url5 = "http://%s:%s/getBaseAnnotation" % (ip, port)
# test_service(url5, data={"anyou": "借贷纠纷_民间借贷", "sentence": sentence})

request_data = {
                "anyou_name":"借贷纠纷_民间借贷",
                "source":"原告诉称",
                "contentHtml":"",
                "insert_data":[{
                        "id": 322213,
                        "content":"",
                        "situation":"存在借款合同",
                        "factor":"存在借款合同",
                        "start_pos":8,
                        "end_pos":20,
                        "pos_or_neg":1,}
                ]}
url6 = "http://%s:%s/insertAnnotationData" % (ip, port)
test_service(url6, data=request_data)