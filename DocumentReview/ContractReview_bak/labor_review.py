#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 15:18
# @Author  : Adolf
# @Site    : 
# @File    : labor_review.py
# @Software: PyCharm
from pprint import pprint, pformat

from DocumentReview.ContractReview_bak.basic_contract import BasicUIEAcknowledgement


class LaborUIEAcknowledgement(BasicUIEAcknowledgement):

    def specific_rule(self, row, extraction_res):
        # if row['pos rule'] == "比对":
        self.id_card_rule(row, extraction_res)
        # self.logger.debug(pformat(extraction_res))
        # if row['schema'] == '联系电话':
        #     if len()
        # exit()
        # pprint(extraction_res)
        # exit()


if __name__ == '__main__':
    acknowledgement = LaborUIEAcknowledgement(config_path="DocumentReview/Config_bak/LaborConfig/labor_20220615.csv",
                                              log_level="debug",
                                              model_path="model/uie_model/labor/model_best")
    acknowledgement.review_main(content="data/DocData/LaborContract/劳动合同.docx", mode="docx")
    pprint(acknowledgement.review_result)
