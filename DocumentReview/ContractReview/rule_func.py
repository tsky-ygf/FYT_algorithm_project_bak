#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 14:06
# @Author  : Adolf
# @Site    : 
# @File    : rule_func.py
# @Software: PyCharm
from id_validator import validator


# 身份证校验
def check_id_card(row, extraction_res, usr):
    res_dict = {}
    if row['schema'] in extraction_res:
        id_card = extraction_res[row["schema"]][0]["text"]
        if validator.is_valid(id_card):
            res_dict["审核结果"] = "通过"
            res_dict["内容"] = id_card
            res_dict["start"] = extraction_res[row["schema"]][0]["start"]
            res_dict["end"] = extraction_res[row["schema"]][0]["end"]
            if usr == "Part A":
                res_dict["法律建议"] = row["A pos legal advice"]
            else:
                res_dict["法律建议"] = row["B pos legal advice"]
        else:
            res_dict["审核结果"] = "不通过"
            res_dict["内容"] = id_card
            res_dict["法律建议"] = row["jiaoyan error advice"]

    elif row['neg rule'] == "未识别，不做审核":
        pass
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = "没有该项目内容"
        res_dict["法律建议"] = row["neg legal advice"]

    return res_dict


# 识别规则
def check_identify(row, extraction_res, usr):
    res_dict = {}
    if row['schema'] in extraction_res:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = extraction_res[row['schema']][0]['text']
        res_dict["start"] = extraction_res[row['schema']][0]['start']
        res_dict["end"] = extraction_res[row['schema']][0]['end']
        if usr == "Part A":
            res_dict["法律建议"] = row["A pos legal advice"]
        else:
            res_dict["法律建议"] = row["B pos legal advice"]
    elif row['neg rule'] == "未识别，不做审核":
        pass
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = "没有该项目内容"
        res_dict["法律建议"] = row["neg legal advice"]

    return res_dict
