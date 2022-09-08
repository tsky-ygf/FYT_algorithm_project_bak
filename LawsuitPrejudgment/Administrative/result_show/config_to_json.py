#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 15:59
# @Author  : Adolf
# @Site    : 
# @File    : config_to_json.py
# @Software: PyCharm
import json
import pandas as pd
from pprint import pprint

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def handle_config(config_csv_path, config_json_path, type_json_path):
    # tax_config = pd.read_csv("LawsuitPrejudgment/Administrative/result_show/tax_config.csv")
    tax_config = pd.read_csv(config_csv_path)
    tax_config.fillna("", inplace=True)

    # print(tax_config)
    # law_dict = {}
    # for index, row in tax_config.iterrows():
    # pprint(row)
    #     law_name = row["法条依据"].split("|")
    #     law_content = row["法条内容"].split("|")
    #     # print(law_name)
    #     # print(law_content)
    #
    #     assert len(law_name) == len(law_content)
    #     for idx, one_law in enumerate(law_name):
    #         if one_law not in law_dict:
    #             law_dict[one_law] = law_content[idx].replace("\u3000", "").replace('\n', '')
    #
    #     # break
    # pprint(law_dict)

    # situation = tax_config["具体情形（来源案例）"].values.tolist()
    # print(situation)
    situation_dict = {}
    # one_situation = tax_config['一级'].values.tolist()
    type_dict = {}
    for index, row in tax_config.iterrows():
        pprint(row)
        if row['一级'] == '':
            continue
        if row['一级'] not in type_dict:
            type_dict[row['一级']] = {}

        if row['二级'] not in type_dict[row['一级']]:
            type_dict[row['一级']][row['二级']] = []

        type_dict[row['一级']][row['二级']].append(row['情形抽取'])
        # break
        situation_dict[row['情形抽取']] = {}
        situation_dict[row['情形抽取']]['问题类别'] = row['一级']
        situation_dict[row['情形抽取']]['法条类别'] = row['二级']
        situation_dict[row['情形抽取']]['法条依据'] = {}

        law_name = row["法条依据"].split("|")
        law_content = row["法条内容"].split("|")
        # print(law_name)
        # print(law_content)

        assert len(law_name) == len(law_content)
        for idx, one_law in enumerate(law_name):
            situation_dict[row['情形抽取']]['法条依据'][one_law] = law_content[idx].replace("\u3000", ""). \
                replace('\n', '')

        situation_dict[row['情形抽取']]['相关案例'] = row['类案'].split('|')
        situation_dict[row['情形抽取']]['处罚依据'] = row['处罚依据'].split('|')

        situation_dict[row['情形抽取']]['处罚种类'] = row['处罚种类'].split('|')
        situation_dict[row['情形抽取']]['处罚幅度'] = row['处罚幅度'].split('|')

        criminal_dict = {}
        situation_dict[row['情形抽取']]['涉刑风险'] = {}
        criminal_name = row['涉刑风险'].split('|')
        criminal_base = row['刑法依据'].split('|')
        criminal_content = row['刑法内容'].split('|')

        assert len(criminal_name) == len(criminal_base) == len(criminal_content)

        for idx, one_criminal in enumerate(criminal_name):
            situation_dict[row['情形抽取']]['涉刑风险'][one_criminal.replace('【', '').replace('】', '')] = [
                criminal_base[idx].replace("\u3000", "").replace('\n', ''),
                criminal_content[idx].replace("\u3000", "").replace('\n', '')]

        # break

    pprint(situation_dict)

    info_json = json.dumps(situation_dict, sort_keys=False, indent=4, ensure_ascii=False)
    # with open("LawsuitPrejudgment/Administrative/result_show/tax_config.json", "w", encoding="utf-8") as f:
    with open(config_json_path, "w", encoding="utf-8") as f:
        f.write(info_json)

    pprint(type_dict)

    type_json = json.dumps(type_dict, sort_keys=False, indent=4, ensure_ascii=False)
    # with open("LawsuitPrejudgment/Administrative/result_show/tax_type.json", "w", encoding="utf-8") as f:
    with open(type_json_path, "w", encoding="utf-8") as f:
        f.write(type_json)


if __name__ == '__main__':
    # department_list = ["tax", "police", "transportation", "port", "other_traffic", "traffic_police"]
    # department_list = ["chengguan", "huazhuang", "market", "shipin","health"]
    # department_list = ["fishery", 'construction']
    # department_list = ['fire', 'travel']
    with open('LawsuitPrejudgment/Administrative/config/supported_administrative_types.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
        department_list = [one_type['type_id'] for one_type in config["supported_administrative_types"]]
        # print(department_list)
    #
    for department in department_list:
        _config_csv_path = "LawsuitPrejudgment/Administrative/config/{}_config.csv".format(department)
        _config_json_path = "data/administrative_config/{}_config.json".format(department)
        _type_json_path = "data/administrative_config/{}_type.json".format(department)
        handle_config(_config_csv_path, _config_json_path, _type_json_path)
