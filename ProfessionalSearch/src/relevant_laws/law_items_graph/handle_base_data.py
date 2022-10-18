#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/4 18:15
# @Author  : Adolf
# @Site    :
# @File    : handle_base_data.py
# @Software: PyCharm
import json
import numpy as np
import pandas as pd
import itertools
from pprint import pprint
from tqdm.auto import tqdm
import re
import json


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


def chuli(old_s):  # 保留中文、大小写、数字
    cop = re.compile("[^\u4e00-\u9fa5^0-9]")  # 匹配不是中文、大小写、数字的其他字符
    nwe_s = cop.sub("", old_s)  # 将old_s中匹配到的字符替换成空s字符
    return nwe_s


def statistics_items_amount(df_path):
    origin_df = pd.read_csv(df_path)
    items_amount_dict = dict()
    old_to_new_dict = old_to_new()
    for i in tqdm(origin_df.index):
        try:
            law_list = origin_df.law_items[i].split("|")
        except Exception as e:
            print(e)
            print(origin_df.law_items[i])
            continue
        # print(law_list)
        law_list = [x for x in law_list if "诉讼" not in x]
        # print(law_list)
        for one_law in law_list:
            law_name_list = re.findall("《(.*?)》", one_law)
            # print(law_name_list)
            for law_name in law_name_list:

                law_name = chuli(law_name)
                if law_name in old_to_new_dict:
                    law_name = old_to_new_dict[law_name]
                # print(law_name)
                # exit()
                if law_name in items_amount_dict:
                    items_amount_dict[law_name] += 1
                else:
                    items_amount_dict[law_name] = 1
            # break
        # break
        # for one_law in law_list:
        #     if one_law not in items_amount_dict:
        #         items_amount_dict[one_law] = 1
        #     else:
        #         items_amount_dict[one_law] += 1

    # statistics_json = json.dumps(items_amount_dict, sort_keys=False, indent=4, separators=(',', ': '))
    # with open("data/law/law_lib/statistics_json.json", "w") as f:
    #     f.write(statistics_json)
    items_amount_res = sorted(
        items_amount_dict.items(), key=lambda x: x[1], reverse=True
    )
    # pprint(items_amount_dict)
    # print(items_amount_res)
    items_amount_dict = dict(items_amount_res)

    pprint(items_amount_dict, sort_dicts=False)

    items_amount_json = json.dumps(
        items_amount_dict, sort_keys=False, indent=4, separators=(",", ": ")
    )

    with open("RelevantLaws/LegalLibrary/law_items_graph/minshi_item.json", "w") as f:
        f.write(items_amount_json)

    print(len(items_amount_dict))
    return items_amount_dict


def old_to_new():
    # print('test')
    df = pd.read_csv("data/law/law_lib/民法典新旧映射表.csv")
    old_to_new_dict = dict()
    for index, row in df.iterrows():
        old_name = row["old_law_name"]
        new_name = row["new_law_name"]
        if pd.isna(old_name) or pd.isna(new_name):
            continue
        old_name = chuli(old_name)
        new_name = chuli(new_name)

        if old_name not in old_to_new_dict:
            old_to_new_dict[old_name] = new_name
    # print(df)
    return old_to_new_dict


if __name__ == "__main__":
    statistics_items_amount(df_path="data/law/law_lib/item_minshi.csv")
    # pprint(old_to_new())
