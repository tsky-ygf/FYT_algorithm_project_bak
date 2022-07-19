#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 13:31
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
# from id_validator import validator

from pprint import pprint, pformat
from DocumentReview.ContractReview import rule_func


class BasicAcknowledgement:
    def __init__(self, config_path, log_level='INFO', *args, **kwargs):
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
        extraction_res = self.check_data_func()
        self.rule_judge(extraction_res[0])

    def check_data_func(self, *args, **kwargs):  # 审核数据
        raise NotImplementedError

    def rule_judge(self, *args, **kwargs):
        raise NotImplementedError

    def read_origin_content(self, content="", mode="text"):
        self.logger.debug("mode: {}".format(mode))
        self.logger.debug("content: {}".format(content))

        if mode == "text":
            content = content.replace(" ", "").replace("\u3000", "")
            text_list = content.split("\n")
        elif mode == "docx":
            text_list = read_docx_file(docx_path=content)
        elif mode == "txt":
            with open(content, encoding='utf-8', mode='r') as f:
                text_list = f.readlines()
                text_list = [line.strip() for line in text_list]
        else:
            raise Exception("mode error")

        self.logger.info("text_list: {}".format(text_list))
        return text_list


class BasicUIEAcknowledgement(BasicAcknowledgement):
    def __init__(self, model_path='', device_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.schema = list(set(self.config['schema'].tolist()))
        self.usr = kwargs.get("usr", "Part A")
        self.schema = self.config['schema'].tolist()
        self.review_result = {schema: {} for schema in self.schema}

        self.data = ""

        if model_path == '':
            self.ie = Taskflow('information_extraction', schema=self.schema, device_id=device_id)
        else:
            self.ie = Taskflow('information_extraction', schema=self.schema, device_id=device_id,
                               task_path=model_path)

        self.logger.info(model_path)

    def check_data_func(self):
        res = self.ie(self.data)
        self.logger.info(pformat(res))
        return res

    def rule_judge(self, extraction_res):
        self.logger.debug("res: {}".format(extraction_res))
        for index, row in self.config.iterrows():
            # self.logger.debug("row: {}".format(row))
            if row["schema"] == "识别":
                rule_res = rule_func.check_identify(row, extraction_res, self.usr)
            elif row["schema"] == "身份证校验":
                rule_res = rule_func.check_id_card(row, extraction_res, self.usr)
            else:
                raise Exception("schema error")
            # result_dict.update(rule_res)
            rule_res['法律依据'] = row['legal basis']
            rule_res['风险等级'] = row['risk level']

            self.review_result[row['schema']].update(rule_res)


if __name__ == '__main__':
    acknowledgement = BasicUIEAcknowledgement(config_path="DocumentReview/Config/fangwuzulin.csv",
                                              log_level="DEBUG",
                                              model_path="model/uie_model/fwzl/model_best",
                                              usr="Part A")
    acknowledgement.review_main(content="data/DocData/Lease/fw_test.docx", mode="docx")
    pprint(acknowledgement.review_result, sort_dicts=False)
