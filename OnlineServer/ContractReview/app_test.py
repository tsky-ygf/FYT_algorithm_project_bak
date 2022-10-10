#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
import requests

t = requests.get("http://127.0.0.1:8112/get_contract_type").json()["result"]
print(t)
