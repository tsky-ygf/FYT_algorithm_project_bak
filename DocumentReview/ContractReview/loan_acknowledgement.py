#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 10:39
# @Author  : Adolf
# @Site    : 
# @File    : loan_acknowledgement.py
# @Software: PyCharm
import pandas as pd
from pprint import pprint

from Utils.logger import get_module_logger
from DocumentReview.ParseFile.parse_word import read_docx_file


class LoanAcknowledgement:
    def __init__(self, config_path, content, mode="text"):
        self.config = pd.read_csv(config_path)
        self.logger = get_module_logger(module_name="LoanAcknowledgement", level="debug")

        self.data_list = self.read_origin_content(content=content, mode=mode)
        self.logger.debug("data_list: {}".format(self.data_list))

    def review_main(self):
        for index, row in self.config.iterrows():
            pprint(row.to_dict())

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
    loan_acknowledgement.review_main()
