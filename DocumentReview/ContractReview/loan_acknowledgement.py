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
from pprint import pformat

from paddlenlp import Taskflow
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
            regular_res = self.check_data_func(row["关键词（正则）"])

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

    def check_data_func(self, *args, **kwargs): # 审核数据
        raise NotImplementedError

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


class LoanUIEAcknowledgement(LoanAcknowledgement):
    def __init__(self, config_path, content, mode="text"):
        super().__init__(config_path=config_path, content=content, mode=mode)
        self.schema = list(set(self.config['schema'].tolist()))
        self.logger.debug("schema: {}".format(self.schema))
        # exit()
        self.ie = Taskflow('information_extraction', schema=self.schema, device_id=1,
                           task_path="model/uie_model/model_best/")

    def check_data_func(self):
        data = '\n'.join(self.data_list)
        res = self.ie(data)
        return res

    def review_main(self):
        review_result = OrderedDict()
        res = self.check_data_func()
        self.logger.debug(res)
        # for index, row in self.config.iterrows():
        #     self.logger.debug(pformat(row.to_dict()))
        #     break

        return review_result


if __name__ == '__main__':
    # loan_acknowledgement = LoanAcknowledgement("DocumentReview/Config/loan.csv", content="data/DocData/IOU.docx",
    #                                            mode="docx")
    loan_acknowledgement = LoanUIEAcknowledgement("DocumentReview/Config/LoanConfig/jietiao_rule_20220525.csv",
                                                  content="data/DocData/IOU.docx",
                                                  mode="docx")
    pprint(loan_acknowledgement.review_main())
