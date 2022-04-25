#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/22 11:51
# @Author  : Adolf
# @Site    : 
# @File    : labor_contract.py
# @Software: PyCharm
from DocumentReview.ParseFile.parse_word import read_docx_file

text_list = read_docx_file(docx_path="data/DocData/LaborContract/劳动合同.docx")

risk_point_list = {"签订日期": "",
                   "签订地点": "",
                   "签订人": "",
                   "签订人职务": "",
                   "签订人职称": "",
                   "签订人职级": "",
                   "签订人职称级别": "",
                }

for index, text in enumerate(text_list):
    print(index, "###", text)
