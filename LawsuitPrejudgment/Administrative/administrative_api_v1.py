#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 14:35
# @Author  : Adolf
# @Site    : 
# @File    : administrative_api_v1.py
# @Software: PyCharm
import json
import pandas as pd


def get_administrative_prejudgment_situation(administrative_type):
    with open('LawsuitPrejudgment/Administrative/result_show/{}_type.json'.format(administrative_type), 'r') as f2:
        type_data = json.load(f2)

    return type_data


def get_administrative_prejudgment_result(administrative_type, situation):
    """
    获取行政处罚的预测结果
    :return:
    """
    # 获取行政处罚的预测结果
    with open('LawsuitPrejudgment/Administrative/result_show/{}_config.json'.format(administrative_type), 'r') as f1:
        info_data = json.load(f1)

    # with open('LawsuitPrejudgment/Administrative/result_show/{}_type.json'.format(administrative_type), 'r') as f2:
    #     type_data = json.load(f2)

    prejudgment_result = dict()
    prejudgment_result["具体情形"] = '{}({})'.format(situation, info_data[situation]['法条类别'])

    prejudgment_result["涉嫌违法行为"] = info_data[situation]['处罚依据']
    prejudgment_result["法条依据"] = info_data[situation]['法条依据']
    prejudgment_result["处罚种类"] = info_data[situation]['处罚种类']
    prejudgment_result["处罚幅度"] = info_data[situation]['处罚幅度']
    prejudgment_result["涉刑风险"] = info_data[situation]['涉刑风险']
    prejudgment_result["相似类案"] = info_data[situation]['相关案例']

    return prejudgment_result


if __name__ == '__main__':
    from pprint import pprint

    # res = get_administrative_prejudgment_result("tax", "无真实业务往来")
    # pprint(res, sort_dicts=False)

    rest = get_administrative_prejudgment_situation("tax")
    pprint(rest, sort_dicts=False)
