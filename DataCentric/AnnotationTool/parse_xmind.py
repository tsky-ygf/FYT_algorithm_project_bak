#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 13:49
# @Author  : Adolf
# @Site    : 
# @File    : parse_xmind.py
# @Software: PyCharm
from pathlib import Path
import pandas as pd
from xmindparser import xmind_to_dict


def trans_xmind_csv(_xmind_file):
    print(_xmind_file)
    anyou_dict_list = xmind_to_dict(_xmind_file)[0]['topic']['topics'][0]['topics']

    case_list = []
    opinion_list = []
    feature_list = []

    for one_case in anyou_dict_list:
        for one_features in one_case['topics']:
            # print(one_features['title'])
            try:
                for one_feature in one_features['topics'][0]['topics'][0]['topics'][0]['topics']:
                    if one_feature['title'] not in ["and", "or", "not"]:
                        case_list.append(one_case['title'])
                        opinion_list.append(one_features['title'])
                        feature_list.append(one_feature['title'])
                    else:
                        for one in one_feature['topics']:
                            case_list.append(one_case['title'])
                            opinion_list.append(one_features['title'])
                            feature_list.append(one['title'])
            except:
                print(one_case['title'])
                # print(one_features['topics'][0]['topics'][0]['topics'][0])
                case_list.append(one_case['title'])
                opinion_list.append(one_features['title'])
                feature_list.append(one_features['topics'][0]['topics'][0]['topics'][0]['title'])

    _anyou_df = pd.DataFrame({'case': case_list, 'opinion': opinion_list, 'feature': feature_list})
    # print(anyou_df)
    return _anyou_df


xmind_file = "data/LawsuitPrejudgment/config/借贷纠纷/借贷纠纷_民间借贷.xmind"
trans_xmind_csv(xmind_file)

xmind_config_path = Path("data/LawsuitPrejudgment/config/")
for xmind_path in xmind_config_path.glob("**/*.xmind"):
    # print(xmind_path)
    # print(xmind_path.name.replace('xmind', 'csv'))
    anyou_df = trans_xmind_csv(xmind_path)
    # print(anyou_df)
    anyou_df.to_csv("data/LawsuitPrejudgment/CaseFeatureConfig/" + xmind_path.name.replace('xmind', 'csv'), index=False)
    # break
