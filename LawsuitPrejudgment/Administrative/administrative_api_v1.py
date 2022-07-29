#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 14:35
# @Author  : Adolf
# @Site    : 
# @File    : administrative_api_v1.py
# @Software: PyCharm
import json
# import pandas as pd
from loguru import logger
from pprint import pprint, pformat


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
    # logger.info(pformat(info_data[situation]))
    prejudgment_result = dict()

    prejudgment_result["specific_situation"] = {
        "title": "具体情形",
        "content": '{}({})'.format(situation, info_data[situation]['法条类别'])
    }
    prejudgment_result["suspected_illegal_act"] = {"title": "涉嫌违法行为", "content": info_data[situation]['处罚依据']}
    prejudgment_result["legal_basis"] = {
        "title": "法条依据",
        "content": [{"law_item": law_item, "law_content": law_content} for law_item, law_content in info_data[situation]['法条依据'].items()]
    }
    prejudgment_result["punishment_type"] = {"title": "处罚种类", "content": info_data[situation]['处罚种类']}
    prejudgment_result["punishment_range"] = {"title": "处罚幅度", "content": info_data[situation]['处罚幅度']}
    prejudgment_result["criminal_risk"] = {
        "title": "涉刑风险",
        "content": [{"crime_name": crime_name, "law_item": law_info[0], "law_content": law_info[1]} for crime_name, law_info in info_data[situation]['涉刑风险'].items()]
    }
    prejudgment_result["similar_case"] = {"title": "相似类案", "content": info_data[situation]['相关案例']}

    return prejudgment_result


if __name__ == '__main__':
    res = get_administrative_prejudgment_result("tax", "没有真实的业务、资金往来")
    pprint(res, sort_dicts=False)

    # rest = get_administrative_prejudgment_situation("tax")
    # pprint(rest, sort_dicts=False)
