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
test_service(url1)

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
# test_service(url4,data={"anyou": "婚姻继承_被继承人债务清偿"})

url5 = "http://%s:%s/getBaseAnnotation" % (ip, port)
# test_service(url5, data={"anyou": "借贷纠纷_民间借贷", "sentence": sentence})

test_data = {
  "anyou_name": "借贷纠纷_民间借贷",
  "source": "本院认为",
  "labelingperson":"朱楠琦",
  "contentHtml": "本院认为，被告王腾向原告杨<span style=\"background-color: red;\" id=\"1651137093355\" class=\"keyword-1651137093355\">晓超借款</span>，原、被告之间形成民间借贷法律关系，借款到期，被告王腾未按约定还本付息，其行为已构成违约，除应承担还本付息的义务外，还应按约定支付罚息及其它费用，故对原告杨晓超要求被告王腾偿还借款本金20000元并支付自2016年12月13日起至本息清偿之日止的利息、罚息的诉讼请求，本院予以支持。根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》第三十条“出借人与借款人既约定了逾期利率，又约定了罚息或者其他费用，出借人可<span style=\"background-color: red;\" id=\"1651137100598\" class=\"keyword-1651137100598\">以选</span>择主张逾期利息、罚息或者其他费用，也可以一并主张，但总计超过年利率24%的部分，人民法院不予支持”的规定，本案中，原告杨晓超主张按月息2%支付利息、罚息、违约金，未超出法律限制性规定，本院予以支持。被告王腾经传票传唤，无正当理由拒不到庭参加诉讼，视为自动放弃诉讼权利，应予缺席判决。",
  "insert_data": [
    {
      "id": "test1",
      "mention": "晓超借款",
      "situation": "本金偿还期限有约定的",
      "factor": [
        "约定本金偿还期限",
        "双方对本金偿还有争议"
      ],
      "start_pos": 13,
      "end_pos": 17,
      "pos_or_neg": 1,
      # "labelingperson":"test",
      "content": "本院认为，被告王腾向原告杨晓超借款，原、被告之间形成民间借贷法律关系，借款到期，被告王腾未按约定还本付息，其行为已构成违约，除应承担还本付息的义务外，还应按约定支付罚息及其它费用，故对原告杨晓超要求被告王腾偿还借款本金20000元并支付自2016年12月13日起至本息清偿之日止的利息、罚息的诉讼请求，本院予以支持。根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》第三十条&ldquo;出借人与借款人既约定了逾期利率，又约定了罚息或者其他费用，出借人可以选择主张逾期利息、罚息或者其他费用，也可以一并主张，但总计超过年利率24%的部分，人民法院不予支持&rdquo;的规定，本案中，原告杨晓超主张按月息2%支付利息、罚息、违约金，未超出法律限制性规定，本院予以支持。被告王腾经传票传唤，无正当理由拒不到庭参加诉讼，视为自动放弃诉讼权利，应予缺席判决。"
    },
    {
      "id": "test2",
      "mention": "以选",
      "situation": "盗用他人名义借款",
      "factor": [
        "被冒用人不知道"
      ],
      "start_pos": 210,
      "end_pos": 212,
      "pos_or_neg": 2,
      # "labelingperson":"test",
      "content": "本院认为，被告王腾向原告杨晓超借款，原、被告之间形成民间借贷法律关系，借款到期，被告王腾未按约定还本付息，其行为已构成违约，除应承担还本付息的义务外，还应按约定支付罚息及其它费用，故对原告杨晓超要求被告王腾偿还借款本金20000元并支付自2016年12月13日起至本息清偿之日止的利息、罚息的诉讼请求，本院予以支持。根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》第三十条&ldquo;出借人与借款人既约定了逾期利率，又约定了罚息或者其他费用，出借人可以选择主张逾期利息、罚息或者其他费用，也可以一并主张，但总计超过年利率24%的部分，人民法院不予支持&rdquo;的规定，本案中，原告杨晓超主张按月息2%支付利息、罚息、违约金，未超出法律限制性规定，本院予以支持。被告王腾经传票传唤，无正当理由拒不到庭参加诉讼，视为自动放弃诉讼权利，应予缺席判决。"
    }
  ]
}
url6 = "http://%s:%s/insertAnnotationData" % (ip, port)
# test_service(url6, data=test_data)