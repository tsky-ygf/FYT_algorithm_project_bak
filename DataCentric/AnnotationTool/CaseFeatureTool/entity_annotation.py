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

def get_base_data_dict():
    pass


# get_data = read_data_from_mysql()
#
# feature = "存在借款合同"
# problem="借贷纠纷_民间借贷"

# for index,row in get_data.iterrows():
#     print(row['f40'])
#     break
