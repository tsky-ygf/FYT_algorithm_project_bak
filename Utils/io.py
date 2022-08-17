#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/1 15:49 
@Desc    : None
"""
import json


def read_json_attribute_value(file_path, attribute_name):
    """
        读取json文件中，指定键的值。
    Args:
        file_path: 文件路径。如"LawsuitPrejudgment/lawsuit_prejudgment/api/fyt_pc_data.json"。
        attribute_name: 要读取的键值，如"problem_disambiguation_dict"。

    Returns:
        json文件中，键值attribute_name对应的value。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        attribute_value = json.load(f).get(attribute_name)
    return attribute_value
