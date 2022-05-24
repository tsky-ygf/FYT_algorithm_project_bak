#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 10:39
# @Author  : Adolf
# @Site    : 
# @File    : loan_acknowledgement.py
# @Software: PyCharm
import re
import pandas as pd
from collections import OrderedDict
from pprint import pprint

from Utils.logger import get_module_logger
from DocumentReview.ParseFile.parse_word import read_docx_file


class LoanAcknowledgement:
    def __init__(self, config_path, content, mode="text"):
        self.config = pd.read_csv(config_path)
        self.config = self.config.fillna("")
        self.logger = get_module_logger(module_name="LoanAcknowledgement", level="debug")

        self.data_list = self.read_origin_content(content=content, mode=mode)
        # self.logger.debug("data_list: {}".format(self.data_list))

    def review_main(self):
        review_result = OrderedDict()
        for index, row in self.config.iterrows():
            # self.logger.debug("row: {}".format(row))
            result_dict = OrderedDict()
            regular_res = self.regular_check_data(row["关键词（正则）"])

            if regular_res != "":
                # self.logger.debug(regular_res)
                result_dict["审核结果"] = self.rule_judge(regular_res, row["规则"])
                result_dict["匹配结果"] = regular_res
                result_dict["风险等级"] = row["风险等级"]
                result_dict["法律建议"] = row["法律建议"].replace("\n", "")
            else:
                result_dict["审核结果"] = "内容缺失"
                result_dict["匹配结果"] = regular_res
                result_dict["风险等级"] = row["风险等级"]
                result_dict["法律建议"] = row["法律建议"].replace("\n", "")
            # break
            review_result[row["合同板块"] + "_" + row["特征（审核点）"]] = result_dict

        return review_result

    def regular_check_data(self, regular_str):
        res = ""
        for data in self.data_list:
            # self.logger.debug("data: {}".format(data))
            # self.logger.debug("regular_str: {}".format(regular_str))
            try:
                res = re.search(regular_str, data).group()
            except:
                pass
            # self.logger.debug("res: {}".format(res))

        return res

    def rule_judge(self, regular_res, rule_str):
        self.logger.debug("匹配关键词 {} ==== 对应规则 {}".format(regular_res, rule_str))
        if rule_str == "":
            return "通过"

    @staticmethod
    def read_origin_content(content="", mode="text"):
        if mode == "text":
            text_list = content.split("\n")
        elif mode == "docx":
            text_list = read_docx_file(docx_path=content)
        else:
            raise Exception("mode error")
        return text_list


if __name__ == '__main__':
    loan_acknowledgement = LoanAcknowledgement("DocumentReview/Config/loan.csv", content="data/DocData/IOU.docx",
                                               mode="docx")
    pprint(loan_acknowledgement.review_main())
