#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/22 09:14
# @Author  : Adolf
# @Site    : 
# @File    : lease_review.py
# @Software: PyCharm
from pprint import pprint, pformat

from DocumentReview.ContractReview_bak.basic_contract import BasicUIEAcknowledgement


class LeaseUIEAcknowledgement(BasicUIEAcknowledgement):

    def specific_rule(self, row, extraction_res):
        pass
        # if row['pos rule'] == "比对":
        #     if row['schema'] == "乙方联系方式":
        # self.logger.debug(pformat(extraction_res))
        # self.logger.debug(pformat(row))
        # exit()


if __name__ == '__main__':
    acknowledgement = LeaseUIEAcknowledgement(config_path="DocumentReview/Config_bak/LeaseConfig/fangwu.csv",
                                              log_level="debug",
                                              model_path="model/uie_model/fwzl/model_best")
    acknowledgement.review_main(content="data/DocData/Lease/fw_test.docx", mode="docx")
    pprint(acknowledgement.review_result)
