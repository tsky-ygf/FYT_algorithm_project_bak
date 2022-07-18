#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 11:30
# @Author  : Adolf
# @Site    : 
# @File    : maimai_review.py
# @Software: PyCharm
from pprint import pprint, pformat

from DocumentReview.ContractReview.basic_contract import BasicUIEAcknowledgement


class BusinessUIEAcknowledgement(BasicUIEAcknowledgement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usr = kwargs.get("usr", None)

    def specific_rule(self, row, extraction_res, pos_legal_advice, neg_legel_advice):
        self.logger.debug(pformat(row))
        self.logger.debug(pformat(extraction_res))
        # exit()
        # if row['pos rule'] == "比对":
        #     if row['schema'] == "乙方联系方式":
        # self.logger.debug(pformat(extraction_res))
        # self.logger.debug(pformat(row))
        # exit()


if __name__ == '__main__':
    acknowledgement = BusinessUIEAcknowledgement(config_path="DocumentReview/Config/BusinessConfig/maimai.csv",
                                                 log_level="info",
                                                 model_path="model/uie_model/fwzl/model_best",
                                                 usr="Part A")
    acknowledgement.review_main(content="data/DocData/Business/maimai-2.txt", mode="txt")
    pprint(acknowledgement.review_result, sort_dicts=False)
