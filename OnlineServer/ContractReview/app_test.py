#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
from pprint import pprint

import requests

inputs = {
  "contract_type_id": "maimai",
  "usr": "party_a",
  "contract_content": """包子馅供货合同
甲方（需方）：梁桂芝
身份证号：141124194012255331
乙方（供货商）：张淑英
身份证号：230621195901087749
为了保护供需双方的合法权益，甲乙双方在平等，自愿，互惠互利，协商一致的基础上，就乙方向甲方供应包子馅一事，达成协议如下：
一、合作关系
甲、乙双方自协议签订之日起形成供需合作伙伴关系。

签署时间：   2022年   3月 1  日
甲方（盖章）：
联系人：梁桂芝
联系方式：13643624505
地址：广西壮族自治区南宁市西夏叶街q座569662号

乙方（盖章）：
联系人：张淑英
联系方式：18203183379
地址：青海省长沙县长寿胡路p座543440号"""
}
print(len(inputs['contract_content']))
t = requests.post("http://127.0.0.1:8112/get_contract_review_result", json=inputs).json()["result"]
pprint(t)
