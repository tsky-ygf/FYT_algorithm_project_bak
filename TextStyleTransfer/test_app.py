#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 13:50
# @Author  : Adolf
# @Site    : 
# @File    : test_app.py
# @Software: PyCharm
import requests

# ip = '172.19.82.199'
ip = "localhost"
port = 7999

url = "http://172.19.82.199:7999/translation"
r = requests.post(url, json={"content": "准予原告中国银行股份有限公司盐城开发区支行撤回诉讼。案件受理费1864元，依法减半收取932元，"
                                        "财产保全费870元，合计人民币1802元，由原告中国银行股份有限公司盐城开发区支行负担。"})
result = r.json()
print(result)