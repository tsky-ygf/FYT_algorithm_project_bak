#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
import requests
import json
response = requests.post("http://127.0.0.1:7000/getNews", json={"category": "税法专栏"})
result = response.json()
print(result)