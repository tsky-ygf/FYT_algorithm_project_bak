#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/12 13:40
# @Author  : Adolf
# @Site    : 
# @File    : feature_extraction.py
# @Software: PyCharm
from pprint import pprint
from paddlenlp import Taskflow

# use_schema = ["伤亡情况", "犯罪行为", "犯罪结果", "犯罪数额", "毒品种类", "毒品数量", "酒精含量"]
# ie = Taskflow('information_extraction', schema=use_schema, device_id=3,
#               task_path='model/uie_model/criminal/xing7/model_best/')
#
#
# def get_xing7_result(text):
#     res = ie(text)
#     # print(res)
#     return res
schema_config = {"theft": {'盗窃触发词': ['总金额', '物品', '地点', '时间', '人物', '行为']}}


def init_extract(criminal_type=""):
    use_schema = schema_config[criminal_type]
    model_path = "model/uie_model/criminal/{}/model_best".format(criminal_type)
    ie = Taskflow('information_extraction', schema=use_schema, device_id=3, task_path=model_path)
    return ie

