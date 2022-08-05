#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 18:15
# @Author  : Adolf
# @Site    : 
# @File    : handle_base_data.py
# @Software: PyCharm
import json
import pandas as pd
import itertools
from pprint import pprint

df = pd.read_csv("data/law/law_lib/item_test.csv")
# demo_df = df[:100]

print(df)

statistics_res_dict = {}
for i in df.index:
    law_list = df.law_items[i].split("|")
    # print(law_list)
    law_list = [x for x in law_list if "诉讼" not in x]
    if len(law_list) < 2:
        continue
    for e in itertools.combinations(law_list, 2):
        # print(e)
        if "#".join(e) in statistics_res_dict:
            statistics_res_dict["#".join(e)] += 1
        elif '#'.join(e[::-1]) in statistics_res_dict:
            statistics_res_dict['#'.join(e[::-1])] += 1
        else:
            statistics_res_dict["#".join(e)] = 1
pprint(statistics_res_dict)

statistics_json = json.dumps(statistics_res_dict, sort_keys=False, indent=4, separators=(',', ': '))
with open("data/law/law_lib/statistics_json.json", "w") as f:
    f.write(statistics_json)
