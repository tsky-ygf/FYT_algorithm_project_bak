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

risk_point_dict = OrderedDict()
risk_point_dict["合同名称"]="",
risk_point_dict["身份信息"]={"甲方": {}, "乙方": {}}
risk_point_dict["具体条款"]=OrderedDict()


item_flag = False
jia_flag = True
break_flag = False


def regular_listr(patten, _text_list):
    res = []
    for _text in _text_list:
        try:
            res.append(re.search(patten, _text).group())
        except:
            pass
    return res

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
# item_content = {k: "".join(v) for k, v in item_content.items() if len(v) > 0}

for key, value in item_content.items():
    title = key.replace(re.findall(r"第.*条", key[:10])[0], "")
    # print(title)
    # print(value)
    if len(re.findall(r"期限", title)) > 0:
        value_str = "".join(value)
        if len(re.findall("无固定期限", value_str)) > 0:
            risk_point_dict["具体条款"]["合同时间"] = "无固定期限"

        else:
            risk_point_dict["具体条款"]["合同时间"] = {}
            htqx = re.search("(合同期|合同)[从自为]\d*年\d*月\d*日(起至|到)\d*年\d*月\d*日", value_str)
            risk_point_dict["具体条款"]["合同时间"]["合同期限具体时间"] = htqx.group()

            htsc_ori = re.search("(合同期|合同).*共\d*[年月日]", value_str)
            htsc = re.search("共\d*[年月日]", htsc_ori.group())
            risk_point_dict["具体条款"]["合同时间"]["合同时长"] = htsc.group()

            syqx = re.search("(试用期|试用)[从自为]\d*年\d*月\d*日(起至|到)\d*年\d*月\d*日", value_str)
            risk_point_dict["具体条款"]["合同时间"]["试用期具体时间"] = htqx.group()

            sysc_ori = re.search("(试用期|试用).*共\d*个?[年月日]", value_str)
            sysc = re.search("共\d*个?[年月日]", sysc_ori.group())
            risk_point_dict["具体条款"]["合同时间"]["试用期时长"] = sysc.group()

    if len(re.findall('工作内容', title)) > 0:
        risk_point_dict["具体条款"]["工作内容"] = {}

        # print(value_str)
        risk_point_dict["具体条款"]["工作内容"]["从事工作"] = regular_listr("从事.*?(岗位|职位|工作)", value)[0]

        if len(regular_listr("(同意|愿意).*(调岗|更换.*岗位|调换.*岗位|调配)", value)) > 0:
            risk_point_dict["具体条款"]["工作内容"]["调换岗位"] = True
        if len(regular_listr("(没有|未).*(同意|愿意).*(不得|不能).*(调岗|更换.*岗位|调换.*岗位)", value)) > 0:
            risk_point_dict["具体条款"]["工作内容"]["调换岗位"] = False

    if len(re.findall('工作地点', title)) > 0:
        risk_point_dict["具体条款"]["工作地点"] = {}

        risk_point_dict["具体条款"]["工作地点"]["工作地"] = regular_listr("(工作地点|工作地|上班地)在?.{2,5}[，。；,;.]", value)[0]

        if len(regular_listr("(同意|愿意).*(更换|调换).*工作地", value)) > 0:
            risk_point_dict["具体条款"]["工作地点"]["调换工作地点"] = True
        if len(regular_listr("(没有|未).*(同意|愿意).*(不得|不能).*(更换|调换).*工作地", value)) > 0:
            risk_point_dict["具体条款"]["工作地点"]["调换工作地点"] = False

    if len(re.findall("劳动保护",title)) > 0:
        # risk_point_dict["具体条款"]["劳动保护"] = {}

        if len(regular_listr("劳动保护|安全防护措施|劳动安全卫生制度", value)) > 0:
            risk_point_dict["具体条款"]["劳动保护"] = True

    if len(re.findall("劳动条件",title)) > 0:
        # risk_point_dict["具体条款"]["劳动条件"] = {}
        if len(regular_listr("劳动条件|卫生条件", value)) > 0:
            risk_point_dict["具体条款"]["劳动条件"] = True

    if len(re.findall("职业危害防护",title)) > 0:
        risk_point_dict["具体条款"]["职业危害防护"] = True

    if len(re.findall("(工作时间|工时制度)",title)) > 0:
        if len(regular_listr("(标准工作制|定时工作制)", value)) > 0:
            risk_point_dict["具体条款"]["工作时间"] = "标准工作制"

    if len(re.findall("劳动报酬",title)) > 0:
        pass


pprint(risk_point_dict)
