#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 09:15
# @Author  : Adolf
# @Site    : 
# @File    : entity_annotation.py
# @Software: PyCharm
import pymysql
import pandas as pd
from pathlib import Path
import requests

pd.set_option('display.max_columns', None)

__all__ = ['get_anyou_list', 'get_case_feature_dict', 'get_base_data_dict']


def get_anyou_list():
    anyou_list = []
    csv_config_path = Path("data/LawsuitPrejudgment/CaseFeatureConfig/")
    for csv_path in csv_config_path.glob("**/*.csv"):
        anyou_list.append(csv_path.name.replace('.csv', ''))
    return anyou_list


# print(get_anyou_list())
def get_case_feature_dict(anyou_name):
    # print(anyou_name)
    anyou_case_feature_dict = {}
    df = pd.read_csv(Path("data/LawsuitPrejudgment/CaseFeatureConfig/") / (anyou_name + ".csv"))
    for index, row in df.iterrows():
        # print(row['case'],row['feature'])
        if row['case'] not in anyou_case_feature_dict:
            anyou_case_feature_dict[row['case']] = []
        anyou_case_feature_dict[row['case']].append(row['feature'])
    return anyou_case_feature_dict


# print(get_case_feature(anyou_name="借贷纠纷_民间借贷"))


def read_data_from_mysql():
    connect_big_data = pymysql.connect(host='172.19.82.227',
                                       user='root', password='Nblh@2022',
                                       db='big_data_ceshi227')

    sql_con = '''
            select f2,f13,f40,f44 from big_data_ceshi227.case_list_original_hetong 
            WHERE f12="民间借贷纠纷" AND f10="判决" AND (LENGTH(f40)>1) limit 100;
           '''

    data = pd.read_sql(sql_con, con=connect_big_data)
    # connect_big_data.close()
    return data


def get_base_data_dict(anyou_name, ):
    connect_big_data = pymysql.connect(host='172.19.82.227',
                                       user='root', password='Nblh@2022',
                                       db='big_data_ceshi227')

    anyou_type, anyou_x = anyou_name.split('_')
    print(anyou_type, anyou_x)
    if anyou_name == "借贷纠纷_民间借贷":
        sql_con = '''
        select f2,f13,f40,f44 from big_data_ceshi227.case_list_original_hetong 
        WHERE f12="民间借贷纠纷" AND f10="判决" AND (LENGTH(f40)>1) limit 1;
        '''
    else:
        raise Exception("暂时不支持该案由")

    data = pd.read_sql(sql_con, con=connect_big_data)
    base_data_dict = {"case_id": data["f2"].values[0],
                      "本院认为": data["f13"].values[0],
                      "原告诉称": data["f40"].values[0],
                      "本院查明": data["f44"].values[0]}
    # print(data)
    return base_data_dict


# print(get_base_data_dict("借贷纠纷_民间借贷"))
def get_base_annotation_dict(anyou_name, sentence):
    print(anyou_name, sentence)
    problem, suqiu = anyou_name.split('_')
    url = "http://172.19.82.199:9500/keyword_feature_matching"
    request_data = {
        "sentence": "2014年6月，我借给了何三宇、冯群华20000元并写了借条，约定月息3%，在2014年10月14日前一次还清，同时谭学民、蔡金花作了担保人。到期后，何三宇、冯群华迟迟不还款，现在我想让他们按照约定，还我本金及利息。",
        "problem": problem,
        "suqiu": suqiu
    }
    r = requests.post(url, json=request_data)
    base_annotation_dict = r.json()

    return base_annotation_dict


print(get_base_annotation_dict(anyou_name="借贷纠纷_民间借贷",
                               sentence="2014年6月，我借给了何三宇、冯群华20000元并写了借条，约定月息3%，"
                                        "在2014年10月14日前一次还清，同时谭学民、蔡金花作了担保人。到期后，何三宇、"
                                        "冯群华迟迟不还款，现在我想让他们按照约定，还我本金及利息。"))
