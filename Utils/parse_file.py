#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:26
# @Author  : Adolf
# @Site    : 
# @File    : parse_file.py
# @Software: PyCharm
import yaml


# 解析模型训练文件
def parse_config_file(config_path):
    with open(config_path, 'r') as f:
        res_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        if "BASE_PATH" in res_dict.keys():
            print(res_dict["BASE_PATH"])
            with open(res_dict["BASE_PATH"], 'r') as f2:
                update_dict = yaml.load(f2.read(), Loader=yaml.FullLoader)
                # print(update_dict)
                res_dict.update(update_dict)
    return res_dict
