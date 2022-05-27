#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 09:16
# @Author  : Adolf
# @Site    : 
# @File    : uie_extraction.py
# @Software: PyCharm
from DocumentReview.ParseFile.parse_word import read_docx_file
from pprint import pprint
from paddlenlp import Taskflow

# text_list = read_docx_file(docx_path="data/DocData/LaborContract/劳动合同.docx")
# text = "\n".join(text_list)
# print(text)

text = "借条\n为购买房产，今收到好友张三（身份证号）以转账方式出借的人民币壹万元整（￥10000.00元），借期拾个月，月利率1%，于××××年××月××日到期时还本付息。、" \
       "逾期未还，则按当期一年期贷款市场报价利率（LPR）的4倍计付逾期利息。"

schema = ['标题' ,'借款人', '借款金额', '借款期限', '借款利率', '还款方式', '还款日期', '逾期利率', '逾期费用']
ie = Taskflow('information_extraction', schema=schema, device_id=1, task_path="model/uie_model/model_best/")
pprint(ie(text))

# dict1 = {"id": 1, "text": "昨天晚上十点加班打车回家58元", "relations": [],
#          "entities": [{"id": 0, "start_offset": 0, "end_offset": 6, "label": "时间"},
#                       {"id": 1, "start_offset": 11, "end_offset": 12, "label": "目的地"},
#                       {"id": 2, "start_offset": 12, "end_offset": 14, "label": "费用"}]}
# dict2 = {"id": 2, "text": "三月三号早上12点46加班，到公司54", "relations": [],
#          "entities": [{"id": 3, "start_offset": 0, "end_offset": 11, "label": "时间"},
#                       {"id": 4, "start_offset": 15, "end_offset": 17, "label": "目的地"},
#                       {"id": 5, "start_offset": 17, "end_offset": 19, "label": "费用"}]}
# dict3 = {"id": 3, "text": "8月31号十一点零四工作加班五十块钱", "relations": [],
#          "entities": [{"id": 6, "start_offset": 0, "end_offset": 10, "label": "时间"},
#                       {"id": 7, "start_offset": 14, "end_offset": 16, "label": "费用"}]}

