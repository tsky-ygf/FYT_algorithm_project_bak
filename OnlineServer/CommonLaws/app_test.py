#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
import requests
import json
response = requests.post("http://127.0.0.1:8149/exampleData", json={"category": "税法专栏"})
# response = requests.post("http://127.0.0.1:8149/getNews", json={"uq_id": "e9b672a97cf1d68e57d1cbcd38f6237f","table_name":"swj_hot_news"})
result = response.json()
print(result)
# print(result)