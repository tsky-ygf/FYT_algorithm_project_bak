#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/22 11:51
# @Author  : Adolf
# @Site    : 
# @File    : labor_contract.py
# @Software: PyCharm
import re
from DocumentReview.ParseFile.parse_word import read_docx_file
from collections import OrderedDict
from pprint import pprint

text_list = read_docx_file(docx_path="data/DocData/LaborContract/劳动合同.docx")

risk_point_dict = {
    "合同名称": "",
    "身份信息": {"甲方": {}, "乙方": {}},
    "具体条款": {},
}
item_flag = False
jia_flag = True
break_flag = False

item_content = OrderedDict()

for index, text in enumerate(text_list):
    # print(index, "###", text)
    if index == 0:
        risk_point_dict["合同名称"] = text

    if break_flag:
        break

    if len(re.findall(r"以下无正文", text)) > 0:
        break_flag = True

    else:
        if text[:3] == "第一条":
            item_flag = True

        if not item_flag:
            # print(text)
            if jia_flag:
                if len(re.findall(r"甲方.{0,6}名称|甲方.{0,6}姓名|公司|集团｜用人单位", text)) > 0:
                    # print(text)
                    risk_point_dict["身份信息"]["甲方"]["单位名称"] = text.split("：")[-1]
                if len(re.findall(r"地址|送达地址|联系地址| .*省.*市.*([区|县])", text)) > 0:
                    risk_point_dict["身份信息"]["甲方"]["单位地址"] = text.split("：")[-1]
                if len(re.findall(r"法定代表人", text)) > 0:
                    risk_point_dict["身份信息"]["甲方"]["法人"] = text.split("：")[-1]
                if len(re.findall(r"甲方.*电话｜联系方式|联系电话", text)) > 0:
                    risk_point_dict["身份信息"]["甲方"]["联系方式"] = text.split("：")[-1]
                if len(re.findall(r"乙方.*名称|乙方.*姓名", text)) > 0:
                    jia_flag = False
                    risk_point_dict["身份信息"]["乙方"]["姓名"] = text.split("：")[-1]
            else:
                if len(re.findall(r"身份证号", text)) > 0:
                    risk_point_dict["身份信息"]["乙方"]["身份证号"] = text.split("：")[-1]
                if len(re.findall(r"出生年月", text)) > 0:
                    risk_point_dict["身份信息"]["乙方"]["出生年月"] = text.split("：")[-1]
                if len(re.findall(r"地址", text)) > 0:
                    risk_point_dict["身份信息"]["乙方"]["通讯地址"] = text.split("：")[-1]
                if len(re.findall(r"乙方.*电话｜联系方式|联系电话", text)) > 0:
                    risk_point_dict["身份信息"]["乙方"]["联系方式"] = text.split("：")[-1]

        else:
            # print(text)
            if len(re.findall(r"第.*条", text[:10])) > 0:
                # print(text)
                item_content[text] = []
            else:
                item_content[list(item_content.keys())[-1]].append(text)

# pprint(item_content)
for key,value in item_content.items():
    print(key)
    print(value)

pprint(risk_point_dict)