#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 16:30
# @Author  : Czq
# @File    : utils.py
# @Software: PyCharm
import pandas as pd


def read_config(config_path):
    config_data = pd.read_csv(config_path, encoding='utf-8', na_values=' ', keep_default_na=False)
    config_list = []
    label2alias = dict()
    _alias2label = dict()

    for line in config_data.values:
        config_list.append(line[0])
        alis = line[1].split('|')
        if alis:
            for ali in alis:
                if ali:
                    _alias2label[ali] = line[0]
        _alias2label[line[0]] = line[0]
    config_list = list(filter(None, config_list))
    return config_list, _alias2label


# 生成所有的通用label， 包含别称
def read_config_to_label(config_path, is_long=False):
    # 读取config，将别称也读为schema
    config_list, _alias2label = read_config(config_path)

    config_list.remove('争议解决')
    config_list.remove('通知与送达')
    config_list.remove('甲方解除合同')
    config_list.remove('乙方解除合同')
    config_list.remove('未尽事宜')
    config_list.remove('金额')

    schema2id = dict()
    id2schema = dict()

    if is_long:
        config_list = ['争议解决', '合同生效', '未尽事宜', '通知与送达', '鉴于条款', '附件', '甲方解除合同', '乙方解除合同']

    for i, c in enumerate(config_list):
        schema2id[c] = i
        id2schema[i] = c

    num_labels = len(config_list)

    # for key, value in _alias2label.items():
    #     if _alias2label[key] in schema2id:
    #         schema2id[key] = schema2id[_alias2label[key]]

    return schema2id, id2schema, num_labels


if __name__ == "__main__":
    pass
