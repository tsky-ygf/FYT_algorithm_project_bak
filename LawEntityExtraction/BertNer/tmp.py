#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 20:05
# @Author  : Adolf
# @Site    : 
# @File    : tmp.py
# @Software: PyCharm
from pprint import pprint
from paddlenlp import Taskflow

schema = ['财产归属'] # Define the schema for entity extraction
ie = Taskflow('information_extraction', schema=schema,device_id=-1)
pprint(ie("婚后男的方父母出资首得到付，夫妻名义贷款还贷，房产证只写男方名，离婚后财产如何分配"))