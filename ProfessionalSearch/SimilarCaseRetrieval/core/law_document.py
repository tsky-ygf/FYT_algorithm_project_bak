#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 23/8/2022 15:16 
@Desc    : None
"""


class LawDocument:
    """ 裁判文书类 """

    def __init__(self, raw_data):
        self.doc_id = raw_data.get("doc_id", "") # f8 文书id
        self.doc_title = raw_data.get("doc_title", "") # f28 案件名称 注意可能是nan
        self.case_number = raw_data.get("case_number", "")  # f3 案号
        self.judge_date = raw_data.get("judge_date", "") # f29 发布日期
        self.province = raw_data.get("province", "")  # f5 省份
        self.court = raw_data.get("court", "") # f10 法院名称
        # 正文内容
        self.text_header = raw_data.get("text_header", "") # f30 文本首部
        self.party_information = raw_data.get("party_information", "") # f31 当事人信息
        self.litigation_record = raw_data.get("litigation_record", "") # f32 诉讼记录
        self.basic_information = raw_data.get("basic_information", "") # f33 案件基本情况
        self.judging_basis = raw_data.get("judging_basis", "") # f34 裁判依据
        self.judging_result = raw_data.get("judging_result", "") # f35 判决结果
        self.end_of_text = raw_data.get("end_of_text", "") # f36 文本尾部

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "case_number": self.case_number,
            "judge_date": self.judge_date,
            "province": self.province,
            "court": self.court,
            "text_header": self.text_header,
            "party_information": self.party_information,
            "litigation_record": self.litigation_record,
            "basic_information": self.basic_information,
            "judging_basis": self.judging_basis,
            "judging_result": self.judging_result,
            "end_of_text": self.end_of_text
        }
