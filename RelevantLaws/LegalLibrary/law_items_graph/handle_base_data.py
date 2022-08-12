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
from tqdm.auto import tqdm


# df = pd.read_csv("data/law/law_lib/item_test.csv")
# # demo_df = df[:100]
#
# print(df)
#
# statistics_res_dict = {}
# for i in df.index:
#     law_list = df.law_items[i].split("|")
#     # print(law_list)
#     law_list = [x for x in law_list if "诉讼" not in x]
#     if len(law_list) < 2:
#         continue
#     for e in itertools.combinations(law_list, 2):
#         # print(e)
#         if "#".join(e) in statistics_res_dict:
#             statistics_res_dict["#".join(e)] += 1
#         elif '#'.join(e[::-1]) in statistics_res_dict:
#             statistics_res_dict['#'.join(e[::-1])] += 1
#         else:
#             statistics_res_dict["#".join(e)] = 1
# pprint(statistics_res_dict)
#
# statistics_json = json.dumps(statistics_res_dict, sort_keys=False, indent=4, separators=(',', ': '))
# with open("data/law/law_lib/statistics_json.json", "w") as f:
#     f.write(statistics_json)

def statistics_items_amount(df_path):
    origin_df = pd.read_csv(df_path)
    items_amount_dict = dict()
    for i in tqdm(origin_df.index):
        try:
            law_list = origin_df.law_items[i].split("|")
            # print(law_list)
            law_list = [x for x in law_list if "诉讼" not in x]
            # print(law_list)
            for one_law in law_list:
                if one_law not in items_amount_dict:
                    items_amount_dict[one_law] = 1
                else:
                    items_amount_dict[one_law] += 1
        except Exception as e:
            print(e)

    # statistics_json = json.dumps(items_amount_dict, sort_keys=False, indent=4, separators=(',', ': '))
    # with open("data/law/law_lib/statistics_json.json", "w") as f:
    #     f.write(statistics_json)
    pprint(items_amount_dict)
    print(len(items_amount_dict))
    return items_amount_dict


if __name__ == '__main__':
    statistics_items_amount(df_path='data/law/law_lib/item_xingshi.csv')
