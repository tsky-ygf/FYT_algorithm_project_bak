#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/08 14:08
# @Author  : Czq
# @File    : utils.py
# @Software: PyCharm
import json
import random
import pandas as pd
import torch
from torch.utils.data import Dataset


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


# 加载train和dev数据
def load_data(path):
    out_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line.strip())
            # out_data.append([json_line['content'],json_line['result_list'],json_line['prompt']])
            out_data.append(json_line)
    return out_data


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
def read_config_to_label(args):
    config_path = 'data/data_src/config.csv'
    # 读取config，将别称也读为schema
    config_list, _alias2label = read_config(config_path)
    return ['争议解决','合同生效','未尽事宜','通知与送达','鉴于条款','附件'], _alias2label


class ReaderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
