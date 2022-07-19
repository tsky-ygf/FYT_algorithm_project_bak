#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 15:52
# @Author  : Adolf
# @Site    : 
# @File    : loan_contract_review.py
# @Software: PyCharm
from pprint import pprint, pformat
from DocumentReview.ContractReview.loan_review import LoanUIEAcknowledgement


class LoanContractUIEAcknowledgement(LoanUIEAcknowledgement):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def specific_rule_add(self, row, extraction_res, pos_legal_advice, neg_legel_advice):
        self.logger.debug(pformat(extraction_res))
        # exit()
        # self.logger.debug(row)
        self.id_card_rule(row, extraction_res, pos_legal_advice, neg_legel_advice)

    # def rule_judge(self, extraction_res,pos_legal_advice, neg_legel_advice):
    #     # self.logger.info(pformat(extraction_res))
    #     for index, row in self.config.iterrows():
    #         # self.logger.debug(pformat(row.to_dict()))
    #         if not self.basic_rule(row, extraction_res,pos_legal_advice, neg_legel_advice):
    #             continue
    #
    #         self.specific_rule(row, extraction_res)
    #         self.specific_rule_add(row, extraction_res)


if __name__ == '__main__':
    acknowledgement = LoanContractUIEAcknowledgement(
        config_path="DocumentReview/Config/LoanConfig/jiekuan_20220605.csv",
        log_level="debug",
        model_path="model/uie_model/jkht/model_best")
    acknowledgement.review_main(content="data/DocData/jkht/jkht.docx", mode="docx")
    pprint(acknowledgement.review_result)
