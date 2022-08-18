#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 16:28
# @Author  : Adolf
# @Site    : 
# @File    : drug_extraction.py
# @Software: PyCharm
from paddlenlp import Taskflow

span_list = ['毒品数量', '毒品种类', '毒品金额']
# use_schema = [{'贩卖': span_list}, {'持有': span_list}, {'容留': span_list}, {'运输': span_list}, {'种植': span_list}]
use_schema = [{'被告人': ['贩卖', '持有', '容留', '运输', '种植']}]
ie = Taskflow('information_extraction', schema=use_schema, device_id=3,
              task_path='model/uie_model/criminal/drug/model_best/')

ie2 = Taskflow('information_extraction', schema=span_list, device_id=3,
               task_path='model/uie_model/criminal/drug/model_best/')


def get_drug_result(text):
    res_relations = ie(text)
    res_span = ie2(text)
    # print(res)
    return res_relations[0], res_span[0]


# result_relations, result_span = get_drug_result(
#     text="公诉机关指控，2015年1月13日22时30分许至23时30分，被告人陈某先后在重庆市江北区北城旺角X栋X楼、负X楼"
#          "附近，两次将共计净重1.33克的海洛因贩卖给左某。公诉机关当庭举示了相应证据证明其指控，据此认为"
#          "被告人陈某的行为触犯了《中华人民共和国刑法》××××、××的规定，已构成贩卖毒品罪，提请对其依法判处。")
#
# print(result_relations)
# print(result_span)
