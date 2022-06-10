#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/9 15:52
# @Author  : Adolf
# @Site    : 
# @File    : loan_contract_review.py
# @Software: PyCharm
from pprint import pprint, pformat
from DocumentReview.ContractReview.loan_review import LoanUIEAcknowledgement

from id_validator import validator


class LoanContractUIEAcknowledgement(LoanUIEAcknowledgement):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def specific_rule_add(self, row, extraction_res):
        self.logger.debug(pformat(extraction_res))
        # exit()
        # self.logger.debug(row)
        if row['schema'] == "身份证号码/统一社会信用代码":
            if "身份证号码/统一社会信用代码" in extraction_res:
                # pass

                id_card = extraction_res[row["schema"]][0]["text"]
                if validator.is_valid(id_card):
                    self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                    self.review_result[row["schema"]]["审核结果"] = "通过"
                    self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]
                else:
                    self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                    self.review_result[row["schema"]]["审核结果"] = "不通过"
                    self.review_result[row["schema"]]["法律建议"] = "输入的身份证账号存在错误"
            else:
                self.review_result[row["schema"]]["内容"] = "未识别到该项内容"
                self.review_result[row["schema"]]["审核结果"] = "不通过"
                self.review_result[row["schema"]]["法律建议"] = row["neg legal advice"]

    def rule_judge(self, extraction_res):
        # self.logger.info(pformat(extraction_res))
        for index, row in self.config.iterrows():
            # self.logger.debug(pformat(row.to_dict()))
            if not self.basic_rule(row, extraction_res):
                continue

            self.specific_rule(row, extraction_res)
            self.specific_rule_add(row, extraction_res)


if __name__ == '__main__':
    acknowledgement = LoanContractUIEAcknowledgement(
        config_path="DocumentReview/Config/LoanConfig/jiekuan_20220605.csv",
        log_level="debug",
        model_path="model/uie_model/jkht/model_best")
    acknowledgement.review_main(content="data/DocData/jkht/jkht.docx", mode="docx")
    pprint(acknowledgement.review_result)
