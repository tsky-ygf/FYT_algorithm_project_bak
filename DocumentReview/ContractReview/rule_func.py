#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 14:06
# @Author  : Adolf
# @Site    : 
# @File    : rule_func.py
# @Software: PyCharm
import re
import datetime
from id_validator import validator
import cn2an


# 身份证校验
def check_id_card(row, extraction_con, res_dict):
    id_card = extraction_con[0]["text"]
    if validator.is_valid(id_card):
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = id_card
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = id_card
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 房屋用途审核
def check_application(row, extraction_con, res_dict):
    app = extraction_con[0]['text']
    if app in ['居住', '办公', '经营', '仓库', '其他']:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = app
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]

    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = app
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 日期关联审核
def check_date_relation(row, extraction_con, res_dict):
    # print(extraction_con)
    # print(extraction_res['租赁期限'])
    if len(extraction_con) == 1:
        # print(extraction_con)
        con = extraction_con[0]['text']
        tmp = re.findall(r'\d+', con)
        tmp = [int(idx) for idx in tmp]
        if len(tmp) == 6:
            # date1 = datetime.datetime(tmp[0], tmp[1], tmp[2])
            # date2 = datetime.datetime(tmp[3], tmp[4], tmp[5])
            # print(date1)
            # print(date2)
            if tmp[2] == tmp[5]:
                res_dict["审核结果"] = "不通过"
                res_dict["内容"] = con
                res_dict["start"] = extraction_con[0]["start"]
                res_dict["end"] = extraction_con[0]["end"]
                res_dict["法律建议"] = "在合同的有效期限内，终止日期应为起始之日的前一天。"
            else:
                res_dict["审核结果"] = "通过"
                res_dict["内容"] = con
                res_dict["start"] = extraction_con[0]["start"]
                res_dict["end"] = extraction_con[0]["end"]
        else:
            res_dict["审核结果"] = "通过"
            res_dict["内容"] = extraction_con[0]['text']
            res_dict["start"] = extraction_con[0]["start"]
            res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = extraction_con[0]['text']
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    # exit()


# 劳动工资审核
def check_wage(row, extraction_con, res_dict):
    # print(extraction_con)
    wage = extraction_con[0]['text']
    # print(wage)
    tmp = re.findall(r'\d+', wage)[0]
    if int(tmp) > 2000:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = tmp
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = tmp
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 试用期工资审核
def check_probation_wage(row, extraction_con, res_dict, wage):
    # print(extraction_con)
    # print(res_dict)
    probation_wage = extraction_con[0]['text']
    tmp = re.findall(r'\d+', probation_wage)[0]
    # print(wage)
    res_dict["内容"] = tmp
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    if float(tmp) >= 0.8 * float(wage):
        res_dict["审核结果"] = "通过"
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 违约金审核
def check_penalty(row, extraction_con, res_dict):
    print(extraction_con)
    # exit()
    pass


# 利率审核
def check_rate(row, extraction_con, res_dict):
    # print(extraction_con)
    rate_text = extraction_con[0]['text']
    res_dict["内容"] = rate_text
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    if 'LPR' in rate_text.upper():
        multiple = re.search("\d+", rate_text).group()
        if float(multiple) > 4:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"
    else:
        if "日利" in rate_text:
            # ir = ir.replace("日利率", "").replace("%", "")
            ir = re.search("\d+", rate_text).group()
            ir = float(ir) * 365
        elif "月利" in rate_text:
            # ir = ir.replace("月利率", "").replace("%", "")
            ir = re.search("\d+", rate_text).group()
            ir = float(ir) * 12
            # self.logger.debug(ir)
        else:
            ir = re.search("\d+", rate_text).group()
            # ir = ir.replace("年利率", "").replace("%", "")

        if float(ir) > 14.8:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"


upper_num = {"壹": "一", "贰": "二", "叁": "三", "肆": "四", "伍": "五", "陆": "六",
             "柒": "七", "捌": "八", "玖": "九", "拾": "十", "佰": "百", "仟": "千",
             "万": "万", "亿": "亿", "兆": "兆"}


# 金额审核
def check_amount_equal(row, extraction_con, res_dict):
    print(extraction_con)
    exit()
    if len(extraction_con) == 1:
        amount = float(re.search("\d+(.\d{2})?", extraction_con[0]['text']).group())
        chinese_amount = "".join(re.findall("[\u4e00-\u9fa5]", extraction_con[0]['text']))
        # chinese_amount.replace("人民币", "")
    else:
        if len(re.findall('\d+(.\d{2})?', extraction_con[0]['text'])) > 0:
            # self.logger.debug()
            amount = extraction_con[0]['text']
            chinese_amount = extraction_con[1]['text']
        else:
            amount = extraction_con[1]['text']
            chinese_amount = extraction_con[0]['text']
        amount = float(re.search("\d+(.\d{2})?", amount).group())
    list_c = list(set(upper_num.keys()) & set(list(chinese_amount)))
    # self.logger.debug(list_c)
    if len(list_c) > 0:
        for c in list_c:
            chinese_amount = chinese_amount.replace(c, upper_num[c])

    output = cn2an.transform(chinese_amount, "cn2an")
    chinese_amount = float(re.search("\d+(.\d{2})?", output).group())

    if chinese_amount == amount:
        if 'list_c' in locals().keys() and len(list_c) == 0:
            res_dict["内容"] = extraction_con[0]["text"]
            res_dict["审核结果"] = "请使用中文大写"
            res_dict["法律建议"] = row["pos legal advice"]
        else:
            res_dict["内容"] = extraction_con[0]["text"]
            res_dict["审核结果"] = "通过"
            res_dict["法律建议"] = row["pos legal advice"]
    else:
        res_dict["内容"] = extraction_con[0]["text"]
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row['neg legal advice']
