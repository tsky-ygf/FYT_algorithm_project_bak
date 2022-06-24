#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 14:48
# @Author  : Adolf
# @Site    : 
# @File    : basic_contract.py
# @Software: PyCharm
import re
import uuid

import pandas as pd
from collections import OrderedDict

from Utils import Logger
from DocumentReview.ParseFile.parse_word import read_docx_file

from paddlenlp import Taskflow
from id_validator import validator

from pprint import pprint, pformat


class BasicAcknowledgement:
    def __init__(self, config_path, log_level):
        self.logger = Logger(name="Contract_{}".format(uuid.uuid1()), level=log_level).logger
        self.logger.info(self.logger.name)
        self.logger.info("log level:{}".format(log_level))
        self.config = pd.read_csv(config_path)
        self.config = self.config.fillna("")

        # self.data_list = self.read_origin_content(content=content, mode=mode)
        self.data_list = []
        self.data = ""
        # self.logger.debug("data_list: {}".format(self.data_list))
        self.review_result = OrderedDict()

    def review_main(self, content, mode):
        self.data_list = self.read_origin_content(content, mode)
        data = '\n'.join(self.data_list)
        self.data = re.sub("[＿_]+", "", data)
        res = self.check_data_func()
        self.rule_judge(res[0])

    def check_data_func(self, *args, **kwargs):  # 审核数据
        raise NotImplementedError

    def rule_judge(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def read_origin_content(content="", mode="text"):
        if mode == "text":
            content = content.replace(" ", "").replace("\u3000", "")
            text_list = content.split("\n")
        elif mode == "docx":
            text_list = read_docx_file(docx_path=content)
        else:
            raise Exception("mode error")
        return text_list


class BasicUIEAcknowledgement(BasicAcknowledgement):
    def __init__(self, model_path='', device_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.schema = list(set(self.config['schema'].tolist()))
        self.schema = self.config['schema'].tolist()
        self.review_result = OrderedDict({schema: {} for schema in self.schema})

        self.data = ""
        # self.ie = Taskflow('information_extraction', schema=self.schema, device_id=1,
        #                    task_path="model/uie_model/model_best/")

        if model_path == '':
            self.ie = Taskflow('information_extraction', schema=self.schema, device_id=device_id)
        else:
            self.ie = Taskflow('information_extraction', schema=self.schema, device_id=device_id,
                               task_path=model_path)

        self.logger.info(model_path)
        # exit()

    def check_data_func(self):
        res = self.ie(self.data)
        return res

    def basic_rule(self, row, extraction_res):
        self.review_result[row["schema"]]["法律依据"] = row["legal basis"]
        self.review_result[row["schema"]]["风险等级"] = row["risk level"]

        if row['pos rule'] == "关键词匹配":
            if len(re.findall(row["pos keywords"], self.data)) > 0:
                # self.logger.debug("pos keywords match")
                self.review_result[row["schema"]]["内容"] = row["pos keywords"]
                self.review_result[row['schema']]["审核结果"] = "通过"
            else:
                self.review_result[row["schema"]]["内容"] = row["schema"]
                self.review_result[row['schema']]["审核结果"] = "没有该项内容"
            self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]

        if row['schema'] not in extraction_res.keys():
            if row["pos rule"] in ["识别", "比对"]:
                self.review_result[row["schema"]]["内容"] = row["schema"]
                self.review_result[row["schema"]]["审核结果"] = "没有该项目内容"
                self.review_result[row["schema"]]["法律建议"] = row["neg legal advice"]

            # if row['neg rule'] == "未识别":
            #     self.review_result[row["schema"]]["内容"] = row["schema"]
            #     self.review_result[row["schema"]]["审核结果"] = "没有该项目内容"
            #     self.review_result[row["schema"]]["法律建议"] = row["neg legal advice"]
            return False

        if row["pos rule"] == "识别":
            self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
            self.review_result[row["schema"]]["审核结果"] = "通过"
            self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]

        if row["pos rule"] == "匹配":
            keyword_list = row["pos keywords"].split("|")
            # if row["schema"] == "标题":
            #     self.logger.debug(pformat(keyword_list))
            #     self.logger.debug(pformat(extraction_res[row["schema"]]))
            # exit()
            if extraction_res[row["schema"]][0]["text"] in keyword_list:
                self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                self.review_result[row["schema"]]["审核结果"] = "通过"
                self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]

        if row["pos rule"] == "正则匹配":
            # if extraction_res[row["schema"]][0]["text"] in keyword_list:
            self.logger.debug(row['schema'])
            self.logger.debug(extraction_res)
            if len(re.findall(row["pos keywords"], extraction_res[row["schema"]][0]["text"])) > 0:
                self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                self.review_result[row["schema"]]["审核结果"] = "通过"
                self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]
            else:
                self.review_result[row["schema"]]["内容"] = row["schema"]
                self.review_result[row["schema"]]["审核结果"] = "没有该项目内容"
                self.review_result[row["schema"]]["法律建议"] = row["neg legal advice"]

        if row["neg rule"] == "匹配":
            neg_keyword_list = row["neg keywords"].split("|")
            legal_advice_list = row["neg legal advice"].split("|")

            if extraction_res[row["schema"]][0]["text"] in neg_keyword_list:
                self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                self.review_result[row["schema"]]["审核结果"] = "不通过"
                if len(legal_advice_list) > 1:
                    self.review_result[row["schema"]]["法律建议"] = legal_advice_list[
                        neg_keyword_list.index(extraction_res[row["schema"]][0]["text"])]
                else:
                    self.review_result[row["schema"]]["法律建议"] = row["neg legal advice"]
        return True

    def specific_rule(self, row, extraction_res):
        raise NotImplementedError

    def rule_judge(self, extraction_res):
        # self.logger.info(pformat(extraction_res))
        for index, row in self.config.iterrows():
            # self.logger.debug(pformat(row.to_dict()))
            if not self.basic_rule(row, extraction_res):
                continue

            self.specific_rule(row, extraction_res)

        self.logger.debug(self.review_result)
        # self.logger.debug(self.schema)
        # return_review_result = {key: self.review_result[key] for key in self.schema}
        # self.logger.debug(pformat(return_review_result))

    def id_card_rule(self, row, extraction_res):
        if row['schema'] == "身份证号码/统一社会信用代码":
            if "身份证号码/统一社会信用代码" in extraction_res or '身份证号' in extraction_res:
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
