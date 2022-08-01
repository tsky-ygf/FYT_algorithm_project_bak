#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/1 15:59 
@Desc    : None
"""
from LawsuitPrejudgment.lawsuit_prejudgment.shared.utils.io import read_json_attribute_value


def test_read_json_attribute_value():
    file_path = "LawsuitPrejudgment/tests/shared/utils/test_io_data.json"
    attribute_name = "test_attribute"

    result = read_json_attribute_value(file_path, attribute_name)
    assert result["测试纠纷"] == "测试合同"
