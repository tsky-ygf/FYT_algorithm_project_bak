#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 15:18
# @Author  : Adolf
# @Site    : 
# @File    : labor_review.py
# @Software: PyCharm
import re
from pprint import pprint
from pprint import pformat

from DocumentReview.ContractReview.basic_contract import BasicUIEAcknowledgement


# import cn2an
#
# upper_num = {"壹": "一", "贰": "二", "叁": "三", "肆": "四", "伍": "五", "陆": "六",
#              "柒": "七", "捌": "八", "玖": "九", "拾": "十", "佰": "百", "仟": "千",
#              "万": "万", "亿": "亿", "兆": "兆"}


class LaborUIEAcknowledgement(BasicUIEAcknowledgement):

    def specific_rule(self, row, extraction_res):
        # if row['pos rule'] == "比对":
        #     pass
        pprint(extraction_res)
        exit()


acknowledgement = LaborUIEAcknowledgement(config_path="DocumentReview/Config/LaborConfig/labor_20220607.csv",
                                          log_level="debug", )

if __name__ == '__main__':
    acknowledgement.review_main(content="data/DocData/LaborContract/劳动合同.docx", mode="docx")
    pprint(acknowledgement.review_result)
