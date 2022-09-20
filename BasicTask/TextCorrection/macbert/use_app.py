#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 18:16
# @Author  : Adolf
# @Site    : 
# @File    : use_app.py
# @Software: PyCharm
import requests
from pprint import pprint

r = requests.post("http://172.19.82.199:6598/macbert_correct", json={"text": "少先队员因该为老人让坐"})
result = r.json()
pprint(result)
