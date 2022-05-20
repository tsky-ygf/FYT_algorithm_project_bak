#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 09:16
# @Author  : Adolf
# @Site    : 
# @File    : uie_extraction.py
# @Software: PyCharm
from DocumentReview.ParseFile.parse_word import read_docx_file
from pprint import pprint
from paddlenlp import Taskflow

text_list = read_docx_file(docx_path="data/DocData/LaborContract/劳动合同.docx")
text = "\n".join(text_list[:11])
# print(text)

schema = ['甲方单位', '单位地址', '法人', '甲方联系方式', '乙方姓名', '乙方地址', '乙方联系方式', '身份证']
ie = Taskflow('information_extraction', schema=schema, device_id=-1)
pprint(ie(text))

dict1 = {"id": 1, "text": "昨天晚上十点加班打车回家58元", "relations": [],
         "entities": [{"id": 0, "start_offset": 0, "end_offset": 6, "label": "时间"},
                      {"id": 1, "start_offset": 11, "end_offset": 12, "label": "目的地"},
                      {"id": 2, "start_offset": 12, "end_offset": 14, "label": "费用"}]}
dict2 = {"id": 2, "text": "三月三号早上12点46加班，到公司54", "relations": [],
         "entities": [{"id": 3, "start_offset": 0, "end_offset": 11, "label": "时间"},
                      {"id": 4, "start_offset": 15, "end_offset": 17, "label": "目的地"},
                      {"id": 5, "start_offset": 17, "end_offset": 19, "label": "费用"}]}
dict3 = {"id": 3, "text": "8月31号十一点零四工作加班五十块钱", "relations": [],
         "entities": [{"id": 6, "start_offset": 0, "end_offset": 10, "label": "时间"},
                      {"id": 7, "start_offset": 14, "end_offset": 16, "label": "费用"}]}
dict3 = {"id": 4, "text": "5月17号晚上10点35分加班打车回家，36块五", "relations": [],
         "entities": [{"id": 8, "start_offset": 0, "end_offset": 13, "label": "时间"},
                      {"id": 1, "start_offset": 18, "end_offset": 19, "label": "目的地"},
                      {"id": 9, "start_offset": 20, "end_offset": 24, "label": "费用"}]}
