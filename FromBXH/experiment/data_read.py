#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 10:25
# @Author  : Adolf
# @Site    : 
# @File    : data_read.py
# @Software: PyCharm
import json

json_path = "/home/fyt/data/1.json"

# json_file = json.load(json_path)'
index = 0
with open(json_path, 'r') as f:
    for line in f.readlines():
        dic = json.loads(line)
        print(dic)
        index += 1
        if index > 20:
            break
