#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 14:35
# @Author  : Adolf
# @Site    : 
# @File    : administrative_api_v1.py
# @Software: PyCharm
import json
# import pandas as pd
from pprint import pprint

from LawsuitPrejudgment.src.common.data_transfer_object.applicable_law_dto import \
    AdministrativeApplicableLawDictCreator
from LawsuitPrejudgment.src.common.data_transfer_object.similar_case_dto import \
    AdministrativeSimilarCaseDictCreator


def get_administrative_prejudgment_situation(administrative_type):
    with open('data/administrative_config/{}_type.json'.format(administrative_type), 'r') as f2:
        type_data = json.load(f2)

    return type_data


def get_administrative_prejudgment_result(administrative_type, situation):
    """
    获取行政处罚的预测结果
    :return:
    """
    # 获取行政处罚的预测结果
    with open('data/administrative_config/{}_config.json'.format(administrative_type), 'r') as f1:
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
        "content": [{"law_item": law_item, "law_content": law_content} for law_item, law_content in
                    info_data[situation]['法条依据'].items()]
    }
    prejudgment_result["punishment_type"] = {"title": "处罚种类", "content": info_data[situation]['处罚种类']}
    prejudgment_result["punishment_range"] = {"title": "处罚幅度", "content": info_data[situation]['处罚幅度']}
    prejudgment_result["criminal_risk"] = {
        "title": "涉刑风险",
        "content": [{"crime_name": crime_name, "law_item": law_info[0], "law_content": law_info[1]} for
                    crime_name, law_info in info_data[situation]['涉刑风险'].items()]
    }
    prejudgment_result["similar_case"] = [
        AdministrativeSimilarCaseDictCreator.create({"title": "相似类案", "content": content}) for content in
        info_data[situation]['相关案例']]
    law_list = prejudgment_result["legal_basis"]["content"] + prejudgment_result["criminal_risk"]["content"]
    # TODO: 去除可能重复的法条
    prejudgment_result["applicable_law"] = [AdministrativeApplicableLawDictCreator.create(law) for law in law_list]
    # TODO: mock judging rule
    prejudgment_result["judging_rule"] = [
        {
            "rule_id": "rule_189",
            "content": "案外人执行异议之诉中，查明涉案款项实体权益属案外人的，应直接判决停止对涉案款项的执行，无须以不当得利另诉。",
            "source": "越律网",
            "source_url": "https://www.sxls.com/gongbao2018.html"
        }
    ]
    return prejudgment_result


if __name__ == '__main__':
    res = get_administrative_prejudgment_result("tax", "没有真实的业务、资金往来")
    pprint(res, sort_dicts=False)

    # rest = get_administrative_prejudgment_situation("tax")
    # pprint(rest, sort_dicts=False)
