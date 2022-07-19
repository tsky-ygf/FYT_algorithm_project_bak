#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 16:36
# @Author  : Adolf
# @Site    : 
# @File    : labor_contract_sz.py
# @Software: PyCharm
import pandas as pd
import re
import subprocess
from docx import Document
import datetime
import sys
from DocumentReview.ParseFile.parse_word import read_docx_file
from pprint import pprint

loc_salary = {}
df = pd.read_csv("DocumentReview/Config_bak/LaborConfig/salary.csv", index_col=0, header=0, encoding="utf-8")
for index, row in df.iterrows():
    loc = row[0]
    salary = row[1]
    if str(loc) == "nan" or str(salary) == "nan":
        continue
    result = [int(i) for i in re.findall("(\d+)元/月", salary)]
    max_salary = max(result)
    loc_salary[loc] = max_salary
del df
print(loc_salary)


# def convert_doc2docx(in_path, out_path):
#     """
#     doc 转 docx
#     """
#     try:
#         subprocess.check_output(["soffice", "--headless", "--invisible", "--convert-to", "docx",
#                                  in_path, "--outdir", out_path])
#         return True
#     except:
#         return False


def get_text_from_docx(path):
    """
    docx文件 提取 文本内容, 不做任何的加工处理
    """
    document = Document(path)
    # 读取每段资料
    texts = [paragraph.text for paragraph in document.paragraphs]
    return texts


def load_review_items(path):
    """

    :param path:
    :return:
    """
    try:
        df = pd.read_excel(path, index_col=None, header=0)
    except:
        df = pd.read_csv(path, index_col=0, header=0, encoding="utf8")

    review_items = []
    for index, row in df.iterrows():
        name = row[0]
        if str(name) == "nan":
            continue
        temp = {
            "name": row[0],
            "desc": row[1],
            "law": row[2],
            "level": row[3],
            "pattern": row[4],
            "mark": row[5],
        }
        review_items.append(temp)
    return review_items


def review_format(text, review_items):
    """
    主要审核要素是否存在，提示风险等级
    """

    def pretty_string(text):
        if str(text) == "nan":
            return ""
        else:
            return str(text).strip()

    #     review_results = []
    level_dict = {"低": 1, "中": 2, "高": 3, }
    review_results = {}
    for review_item in review_items:
        temp = re.search(review_item["pattern"], text)
        if temp:
            review_results[review_item["name"]] = {
                "risk_level": level_dict.get(review_item["level"].strip()),
                "level": 0,
                "pattern": review_item["pattern"],
                "match": temp.group(),
                "review_result": {
                    "desc": pretty_string(review_item["desc"]),
                    "law": pretty_string(review_item["law"]),
                }
            }
        else:
            review_results[review_item["name"]] = {
                "risk_level": level_dict.get(review_item["level"].strip()),
                "level": level_dict.get(review_item["level"].strip()),
                "pattern": review_item["pattern"],
                "match": "",
                "review_result": {
                    "desc": pretty_string(review_item["desc"]),
                    "law": pretty_string(review_item["law"]),
                }
            }
    return review_results


def str2time(text):
    """
    抽取时间
    """
    result = re.search("(\d{2,4})年(\d{1,2})月(\d{1,2})日", text)
    if result:
        return datetime.datetime(int(result.group(1)), int(result.group(2)), int(result.group(3)))
    else:
        return None


def extract_items(text, contract_item_patterns=[]):
    """
    根据模板抽取要素
    """
    text = re.sub("\s+", "", text)
    text = re.sub("\n+", " ", text)
    text = re.sub("[＿_]+", "＿", text)
    items = {}
    for pattern in contract_item_patterns:
        result = re.search(pattern["pattern"], text)
        if result:
            for _ in pattern["extract"].split("\n"):
                try:
                    temp = re.split("-|—", _)
                    items[temp[0]] = result.group(int(temp[1]))
                except:
                    print(_, "location or format error")
    return items


# ---------------------- 内容审核 ----------------------
def check_rule(text, patterns=None, result=None):
    """
    审核 鉴于条款法律法规
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 低
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL2,
        "review_result": {
            "law": "根据《中华人民共和国劳动法》、《中华人民共和国劳动合同法》以及有关法律、法规、规章和政策的规定，经双方平等协商，用人单位同意聘请劳动者为合同制职工，并订立本合同。双方在签订本合同前，已认真阅读本合同，本合同一经签订，即具有法律效力，双方必须严格履行。"
        }
    } if result is None else result
    patterns = [{
        "pattern": "根据.{2,30}规定",
        "extract": "鉴于条款法律法规-0"
    }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])
    if not items.get("鉴于条款法律法规"):
        result["level"] = LEVEL2
        result["review_result"]["desc"] = "缺失有关法律、法规、规章和政策的规定"
        return {"鉴于条款法律法规": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "审核通过"
        return {"鉴于条款法律法规": result}


def check_period(text, patterns=None, result=None):
    """
    审核试用期条款
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 低
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL3,
        "review_result": {
            "law": "1、劳动合同期限三个月以上不满一年的，试用期不得超过一个月；超过按一个月计算;2、劳动合同期限一年以上不满三年的，试用期不得超过二个月；超过按二个月计算；\n3、三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月；超过按六个月计算。"
        }
    } if result is None else result

    patterns = [
        {
            "pattern": "[^无]{5}(劳动合同|固定期限|合同.{0,2}期限|合同.{0,2}有效期|服务期).{0,9}(自|：|期限|从)([^年自从]{0,4}年.{0,2}月.{0,2}日)(起?至|到)(.{0,4}年.{0,2}月.{0,2}日)",
            "extract": "合同起始日期-3\n合同结束日期-5"
        },
        {
            "pattern": "(劳动合同|固定期限|合同.{0,2}期限|合同.{0,2}有效期|合同).{0,9}(自|于|从)([^年自从]{0,4}年.{0,2}月.{0,2}日)(生效).{0,15}[自于从]([^年自从]{0,4}年.{0,2}月.{0,2}日)终止",
            "extract": "合同起始日期-3\n合同结束日期-5"
        },
        {
            "pattern": "(试用期).{0,9}(自|：|期限|从)([^年自从]{0,4}年.{0,2}月.{0,2}日)(起?至|到)(.{0,4}年.{0,2}月.{0,2}日)",
            "extract": "试用期起始日期-3\n试用期结束日期-5"
        },
        {
            "pattern": "([^年自从]{4}年.{0,2}月.{0,2}日)(起?至|到)(.{0,4}年.{0,2}月.{0,2}日).{0,2}[为是].{0,2}(试用期)",
            "extract": "试用期起始日期-1\n试用期结束日期-3"
        },
        {
            "pattern": "(试用期)[为是]?(.{0,3}月|半年)",
            "extract": "试用期期限-2"
        }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    def get_period_1():
        """
        合同期限计算,以年为单位
        """
        if items.get("合同起始日期"):
            date_1 = str2time(items.get("合同起始日期"))
            if date_1:
                if items.get("合同结束日期"):
                    date_2 = str2time(items.get("合同结束日期"))
                    return (date_2 - date_1).days / 365
                else:
                    return "无固定期限合同"
            else:
                return "未知"
        else:
            if items.get("合同期限"):
                temp = re.search("(\d+).{0,2}年|(\d+).{0,2}月", items.get("合同期限"))
                if temp:
                    year = temp.group(1)
                    mouth = temp.group(2)
                    return int(year) if year else int(mouth) / 12
                else:
                    return "未知"
            else:
                return None

    def get_period_2():
        """
        试用期期限计算，以月为单位
        """
        if items.get("试用期起始日期") or items.get("合同起始日期"):
            try:
                date_1 = str2time(items.get("试用期起始日期"))
            except:
                date_1 = str2time(items.get("合同起始日期"))
            if date_1:
                if items.get("试用期结束日期"):
                    date_2 = str2time(items.get("试用期结束日期"))
                    return ((date_2 - date_1).days + 1) / 30
                else:
                    # 看看是不是已经有期限了
                    if items.get("试用期期限"):
                        temp = re.search("(\d+).{0,2}月", items.get("试用期期限"))
                        if temp:
                            mouth = temp.group(1)
                            return float(mouth)
                        else:
                            temp = re.search("半年", items.get("试用期期限"))
                            if temp:
                                return 6.0
                            return "未知"
                    return "未知"
            else:
                if items.get("试用期期限"):
                    temp = re.search("(\d+).{0,2}月", items.get("试用期期限"))
                    if temp:
                        mouth = temp.group(1)
                        return float(mouth)
                    else:
                        return "未知"
                return "未知"
        else:
            if items.get("试用期期限"):
                temp = re.search("(\d+).{0,2}月", items.get("试用期期限"))
                if temp:
                    mouth = temp.group(1)
                    return float(mouth)
                else:
                    temp = re.search("半年", items.get("试用期期限"))
                    if temp:
                        return 6.0
                    return "未知"
            else:
                return None

    date_1 = get_period_1()
    date_2 = get_period_2()

    # print("合同期限(年)：",date_1)
    # print("试用期期限(月)：",date_2)

    if date_2:
        if isinstance(date_2, str):
            result["level"] = LEVEL0
            result["review_result"]["desc"] = "试用期-未填写"
            return {"试用期条款-期限": result}
        elif isinstance(date_2, float):
            if date_1:
                if isinstance(date_1, str):
                    if date_1 == "无固定期限合同":
                        if date_2 > 6:
                            result["level"] = LEVEL3
                            result["review_result"]["desc"] = "三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月；超过按六个月计算"
                            return {"违法约定试用期期限": result}
                        else:
                            result["level"] = LEVEL0
                            result["review_result"]["desc"] = "审核通过"
                            return {"违法约定试用期期限": result}
                    elif date_1 == "未知":
                        result["level"] = LEVEL0
                        result["review_result"]["desc"] = "合同期限-未填写"
                        return {"违法约定试用期期限": result}
                else:
                    if date_1 < 0.25 and date_2 > 0:
                        result["level"] = LEVEL3
                        result["review_result"]["desc"] = "劳动合同期不满三个月，不设试用期"
                        return {"违法约定试用期期限": result}
                    elif date_1 < 1 and date_2 > 1:
                        result["level"] = LEVEL3
                        result["review_result"]["desc"] = "劳动合同期限三个月以上不满一年的，试用期不得超过一个月；超过按一个月计算"
                        return {"违法约定试用期期限": result}
                    elif 1 <= date_1 < 3 and date_2 > 2:
                        result["level"] = LEVEL3
                        result["review_result"]["desc"] = "劳动合同期限一年以上不满三年的，试用期不得超过二个月；超过按二个月计算"
                        return {"违法约定试用期期限": result}
                    elif 3 <= date_1 and date_2 > 6:
                        result["level"] = LEVEL3
                        result["review_result"]["desc"] = "三年以上固定期限和无固定期限的劳动合同，试用期不得超过六个月；超过按六个月计算"
                        return {"违法约定试用期期限": result}
                    else:
                        result["level"] = LEVEL0
                        result["review_result"]["desc"] = "审核通过"
                        return {"违法约定试用期期限": result}
            else:
                result["level"] = LEVEL0
                result["review_result"]["desc"] = "没有约定合同期限"
                return {"违法约定试用期期限": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "没有约定试用期"
        return {"违法约定试用期期限": result}


def check_salary(text, patterns=None, result=None, min_salary=3000):
    """
    审核 基本工资
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 底
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL3,
        "review_result": {
            "law": "劳动者的工资低于用人单位所在地的最低工资标准的，按用人单位所在地的最低工资标准计算"
        }
    } if result is None else result

    patterns = [
        {
            "pattern": "支付\d{3,8}(工资|薪酬|报酬)",
            "extract": "基本工资-0"
        },
        {
            "pattern": "[^试收]{5}(工资|薪酬|报酬)[为是：:]?([^，；。]{0,8}\d+)[元，；。]",
            "extract": "基本工资-0"
        }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    if items.get("基本工资"):
        temp = re.search("(\d+).*元?", items.get("基本工资"))
        if temp:
            salary = temp.group(1)
            if int(salary) < min_salary:
                result["level"] = LEVEL3
                result["review_result"]["desc"] = "劳动者的工资低于用人单位所在地的最低工资标准的，按用人单位所在地的最低工资标准计算"
                return {"违法约定工资数额": result}
            else:
                result["level"] = LEVEL0
                result["review_result"]["desc"] = "审核通过"
                return {"违法约定工资数额": result}
        else:
            result["level"] = LEVEL0
            result["review_result"]["desc"] = "劳动工资未填写"
            return {"违法约定工资数额": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "未约定劳动工资"
        return {"违法约定工资数额": result}


def check_trial_salary(text, patterns=None, result=None, min_salary=3000):
    """
    审核试用期工资
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 底
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL3,
        "review_result": {
            "law": "劳动者在试用期的工资不得低于本单位相同岗位最低档工资或者劳动合同约定工资的百分之八十，低于上述数额的按上述数额计算，并不得低于用人单位所在地的最低工资标准，低于用人单位所在地的最低工资标准的，按用人单位所在地的最低工资标准计算。"
        }
    } if result is None else result

    patterns = [
        {
            "pattern": "支付\d{3,8}(工资|薪酬|报酬)",
            "extract": "基本工资-0"
        },
        {
            "pattern": "[^试收]{5}(工资|薪酬|报酬)[为是：:]?([^，；。]{0,8}\d+)[元，；。]",
            "extract": "基本工资-0"
        },
        {
            "pattern": "[试][^，；。\d]{0,5}(工资|薪酬|报酬)?[为是：:]?([^，；。\d]{0,8}\d+)[元，；。]",
            "extract": "试用期工资-0"
        }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    if items.get("试用期工资"):
        temp = re.search("(\d+).*元?", items.get("试用期工资"))
        if temp:
            trial_salary = temp.group(1)
            if items.get("基本工资"):
                temp = re.search("(\d+).*元?", items.get("基本工资"))
                if temp:
                    salary = temp.group(1)
                    # print(salary,trial_salary)
                    if int(trial_salary) < int(salary) * 0.8 or int(trial_salary) < min_salary:
                        result["level"] = LEVEL3
                        result["review_result"][
                            "desc"] = "劳动者在试用期的工资不得低于本单位相同岗位最低档工资或者劳动合同约定工资的百分之八十，低于上述数额的按上述数额计算，并不得低于用人单位所在地的最低工资标准，低于用人单位所在地的最低工资标准的，按用人单位所在地的最低工资标准计算。"
                        return {"违法约定试用期工资数额": result}
                    else:
                        result["level"] = LEVEL0
                        result["review_result"]["desc"] = "审核通过"
                        return {"违法约定试用期工资数额": result}
                else:
                    result["level"] = LEVEL0
                    result["review_result"]["desc"] = "劳动工资未填写"
                    return {"违法约定试用期工资数额": result}
            else:
                result["level"] = LEVEL0
                result["review_result"]["desc"] = "未约定劳动工资"
                return {"违法约定试用期工资数额": result}
        else:
            result["level"] = LEVEL0
            result["review_result"]["desc"] = "试用期工资未填写"
            return {"违法约定试用期工资数额": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "未约定试用期工资"
        return {"违法约定试用期工资数额": result}


def check_deduction(text, patterns=None, result=None):
    """
    审核 扣除工资
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 底
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL3,
        "review_result": {
            "law": "企业不得克扣员工的工资。如果由于员工原因给企业造成损失的，企业有权从劳动者工资中扣除部分作为赔偿，但是每月扣除部分不能超过员工当月工资的20%，若扣除后剩余工资部分低于当地最低工资标准，则按最低工资标准支付。"
        }
    } if result is None else result

    patterns = [{
        "pattern": "扣[^，；。]{0,20}工资",
        "extract": "违法约定扣除工资事由-0"
    }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    if items.get("违法约定扣除工资事由"):
        result["level"] = LEVEL3
        result["review_result"][
            "desc"] = "企业不得克扣员工的工资。如果由于员工原因给企业造成损失的，企业有权从劳动者工资中扣除部分作为赔偿，但是每月扣除部分不能超过员工当月工资的20%，若扣除后剩余工资部分低于当地最低工资标准，则按最低工资标准支付。"
        return {"违法约定扣除工资事由": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "审核通过"
        return {"违法约定扣除工资事由": result}


def check_competition(text, patterns=None, result=None):
    """
    审核 竞业限制条款
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 底
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL2,
        "review_result": {
            "law": "竞业限制的期限最长不得超过二年"
        }
    } if result is None else result

    patterns = [
        {
            "pattern": "((一|两|二|1|2).{0,2}年|\D{2}[01]?[1-9]\D{0,2}月|\D{2}[2]?[1-4]\D{0,2}月).{0,55}(竞业|禁止)",
            "extract": "竞业期限2-0"
        },
        {
            "pattern": "(竞业|禁止).{0,8}(一|两|二|1|2).{0,2}年",
            "extract": "竞业期限2-0"
        },
        {
            "pattern": "(竞业|禁止).{0,8}(三|四|五|六|七|八|九|3|4|5|6|7|8|9).{0,2}年",
            "extract": "竞业期限2+-0"
        },
        {
            "pattern": "((三|四|五|六|七|八|九|3|4|5|6|7|8|9).{0,2}年|\D{2}[2-9][0-9]\D{0,2}月).{0,55}(竞业|禁止)",
            "extract": "竞业期限2+-0"
        }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    if items.get("竞业期限2+"):
        result["level"] = LEVEL2
        result["review_result"]["desc"] = "竞业限制的期限最长不得超过二年"
        return {"违法约定竞业限制期限": result}
    elif items.get("竞业期限2"):
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "通过"
        return {"违法约定竞业限制期限": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "未约定竞业期限"
        return {"违法约定竞业限制期限": result}


def check_provision_1(text, patterns=None, result=None):
    """
    审核 违反法律强制性规定
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 底
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL3,
        "review_result": {
            "law": "企业不得以各种形式收取劳动者定金、保证金或抵押金"
        }
    } if result is None else result

    patterns = [{
        "pattern": "(定金|保证金|抵押金)",
        "extract": "违反法律强制性规定1-1"
    }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    if items.get("违反法律强制性规定1"):
        result["level"] = LEVEL3
        result["review_result"]["desc"] = "未约定竞业期限"
        return {"违法收取费用": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "通过"
        return {"违法收取费用": result}


def check_provision_2(text, patterns=None, result=None):
    """
    审核 违反法律强制性规定
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 底
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL3,
        "review_result": {
            "law": "劳动合同的更改需要企业与劳动者双方协商一致"
        }
    } if result is None else result

    patterns = [{
        "pattern": "(单方调整|随时调整|单方更改|随时更改)",
        "extract": "违反法律强制性规定2-1"
    }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    if items.get("违反法律强制性规定2"):
        result["level"] = LEVEL3
        result["review_result"]["desc"] = "劳动合同的更改需要企业与劳动者双方协商一致"
        return {"违法单方更改劳动合同": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "通过"
        return {"违法单方更改劳动合同": result}


def check_provision_3(text, patterns=None, result=None):
    """
    审核 违反法律强制性规定
    """
    LEVEL0 = 0  # 通过
    LEVEL1 = 1  # 底
    LEVEL2 = 2  # 中
    LEVEL3 = 3  # 高
    result = {
        "risk_level": LEVEL3,
        "review_result": {
            "law": "劳动合同法已将约定终止的情形取消，所以不得在劳动合同中将“末位淘汰”约定为终止条件。"
        }
    } if result is None else result

    patterns = [{
        "pattern": "(考核.{0,15}合同.{0,15}终止|末位淘汰)",
        "extract": "违反法律强制性规定3-1"
    }] if patterns is None else patterns

    items = extract_items(text, patterns)

    result["match"] = "\n".join([k + ":" + v for k, v in items.items()])

    if items.get("违反法律强制性规定3"):
        result["level"] = LEVEL3
        result["review_result"]["desc"] = "劳动合同法已将约定终止的情形取消，所以不得在劳动合同中将“末位淘汰”约定为终止条件。"
        return {"违法约定劳动合同终止情形": result}
    else:
        result["level"] = LEVEL0
        result["review_result"]["desc"] = "通过"
        return {"违法约定劳动合同终止情形": result}


# ---------------------- 内容审核 ----------------------
def review_content(text, review_results={}, loc="浙江省"):
    """
    审核 合同内容
    """
    print("review_content...")
    review_results.update(check_rule(text))
    review_results.update(check_period(text))
    review_results.update(check_salary(text, min_salary=loc_salary[loc]))
    review_results.update(check_trial_salary(text, min_salary=loc_salary[loc]))
    review_results.update(check_deduction(text))
    review_results.update(check_competition(text))
    review_results.update(check_provision_1(text))
    review_results.update(check_provision_2(text))
    review_results.update(check_provision_3(text))
    return review_results


def score_review_results(review_results):
    """
    对审核结果进行打分：
    假设：高风险权重=3
    中风险权重=1
    整体风险分=（未通过的高风险审查项*3+未通过的中风险审查项*1）/（所有高风险审查项*3+所有中风险审查项*1）*100%
    """
    risk_score = {
        3: 3,
        2: 1,
        1: 0,
        0: 0
    }
    total_item_count = len(review_results)
    pass_item_count = 0
    risk_item_count_1 = 0
    risk_item_count_2 = 0
    risk_item_count_3 = 0
    total_risk = 0
    now_risk = 0
    for key, value in review_results.items():
        if value["level"] == 0:
            pass_item_count += 1
        elif value["level"] == 1:
            risk_item_count_1 += 1
        elif value["level"] == 2:
            risk_item_count_2 += 1
        elif value["level"] == 3:
            risk_item_count_3 += 1
        else:
            print("some error occur")
        # 打分统计计算
        if value["risk_level"] >= 2:
            total_risk += risk_score[value["risk_level"]]
            now_risk += risk_score[value["level"]]
    print("pass_item_count:", pass_item_count)
    # print("risk_item_count:", risk_item_count)
    print("total_risk:", total_risk)
    print("now_risk:", now_risk)
    print("score:%.2f " % (round(100 - float(now_risk) / total_risk * 100, 2)))
    return {
        "pass_item_count": pass_item_count,
        "risk_item_count_1": risk_item_count_1,
        "risk_item_count_2": risk_item_count_2,
        "risk_item_count_3": risk_item_count_3,
        "total_item_count": total_item_count,
        "score": str(round(100 - float(now_risk) / total_risk * 100, 2))
    }


def main(docx_path,loc="浙江省"):
    review_items = load_review_items("DocumentReview/Config_bak/LaborConfig/review_labor_sz.csv")
    text_list = read_docx_file(docx_path=docx_path)

    text = "\n".join(text_list)
    text = re.sub("\s+", "  ", text)
    text = re.sub("[＿_]+", "＿", text)
    # 使用 规则进行 合同

    # 合同格式审核
    review_results = review_format(text, review_items)
    review_results = review_content(text, review_results, loc=loc)

    # 通过的内容不进行审核
    remove_pass_review_results = {}
    for key, value in review_results.items():
        if value["level"] == 0:
            value["pattern"] = ""
            value["review_result"] = {}
        remove_pass_review_results[key] = value

    # 合同审核打分
    score_result = score_review_results(review_results)

    pprint(review_results)
    pprint(score_result)
    return review_results, score_result


if __name__ == '__main__':
    doc_path = "data/DocData/LaborContract/劳动合同_hl.docx"
    # 合同处理
    loc = "广东省"
    main(doc_path, loc=loc)
