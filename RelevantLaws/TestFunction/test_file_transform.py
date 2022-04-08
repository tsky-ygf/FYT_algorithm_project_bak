#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 15:51
# @Author  : Adolf
# @Site    : 
# @File    : test_file_transform.py
# @Software: PyCharm
from RelevantLaws.Tools.file_transform import get_data, transform_data_2

file_path = "data/law/中华人民共和国个人独资企业法(FBM-CLI-1-23175).docx"
docx_content_list = get_data(file_path)
df = transform_data_2(docx_content_list)
