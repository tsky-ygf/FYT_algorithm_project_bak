#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 10:26
# @Author  : Adolf
# @Site    : 
# @File    : tax_v0.1.py
# @Software: PyCharm
import re

import pandas as pd
from pprint import pprint, pformat

pd.set_option('display.max_columns', None)

tax_config = pd.read_csv("LawsuitPrejudgment/Administrative/tax_config.csv")
tax_data = pd.read_csv("data/DocData/tax_data.csv")

tax_config = tax_config[['案由|行为', '法条']]
tax_config_dict = tax_config.set_index('案由|行为').to_dict()['法条']
tax_config_list = tax_config_dict.keys()

tax_data['event_text'] = tax_data['event_text'].fillna("")


# pprint(tax_config_dict)
# print(tax_data.head())
def tax_extract(tax_text):
    res_tax_name = []
    for _tax_name in tax_config_list:
        if len(re.findall(_tax_name, tax_text)) > 0:
            res_tax_name.append(_tax_name)
    return res_tax_name


count = 0
for index, row in tax_data.iterrows():
    # pprint(row)
    if row['event_text'] == "":
        continue
    event_text = row['event_text']
    # print(event_text)
    tax_name = tax_extract(event_text)
    # print(tax_name)
    if len(tax_name) == 0:
        print(event_text)
        count += 1
        print('-' * 100)
    # break

print(count)
