#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 13:31
# @Author  : Adolf
# @Site    :
# @File    : basic_contract.py
# @Software: PyCharm
import re
# import uuid

import pandas as pd
from collections import OrderedDict

from Utils import Logger
from DocumentReview.ParseFile.parse_word import read_docx_file

from paddlenlp import Taskflow
# from id_validator import validator

from pprint import pprint, pformat
from DocumentReview.ContractReview import rule_func
from DocumentReview.UIETool.deploy.uie_predictor import UIEPredictor

from Utils.logger import print_run_time


class BasicAcknowledgement:
    def __init__(self, config_path, log_level='INFO', *args, **kwargs):
        self.logger = Logger(name="ContractReview", level=log_level).logger
        self.logger.info(self.logger.name)
        self.logger.info("log level:{}".format(log_level))
        self.config = pd.read_csv(config_path)
        self.config = self.config.fillna("")

        # self.data_list = self.read_origin_content(content=content, mode=mode)
        self.data_list = []
        self.data = ""
        self.usr = None
        # self.logger.debug("data_list: {}".format(self.data_list))
        self.review_result = OrderedDict()

    @print_run_time
    def review_main(self, content, mode, usr="Part A"):
        self.review_result = self.init_review_result()
        self.data_list = self.read_origin_content(content, mode)
        data = '\n'.join(self.data_list)
        self.data = re.sub("[＿_]+", "", data)
        extraction_res = self.check_data_func()
        self.usr = usr
        self.rule_judge(extraction_res[0])

        self.review_result = {key: value for key, value in self.review_result.items() if value != {}}

    def init_review_result(self):
        raise NotImplementedError

    def check_data_func(self, *args, **kwargs):  # 审核数据
        raise NotImplementedError

    def rule_judge(self, *args, **kwargs):
        raise NotImplementedError

    def read_origin_content(self, content="", mode="text"):
        # self.logger.debug("mode: {}".format(mode))
        # self.logger.debug("content: {}".format(content))

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

        self.logger.debug("text_list: {}".format(text_list))

        return text_list


class InferArgs:
    model_path_prefix = ""
    position_prob = 0.5
    max_seq_len = 512
    batch_size = 4
    device = "cpu"
    schema = []


class BasicUIEAcknowledgement(BasicAcknowledgement):
    def __init__(self, model_path='', device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.schema = list(set(self.config['schema'].tolist()))
        self.device = device
        self.schema = self.config['schema'].tolist()
        # self.review_result = {schema: {} for schema in self.schema}

        self.data = ""
        if self.device == "cpu":
            args = InferArgs()
            args.model_path_prefix = model_path
            args.schema = self.schema[:3]
            self.predictor = UIEPredictor(args)
        else:
            if model_path == '':
                self.ie = Taskflow('information_extraction', schema=self.schema, device_id=int(device))
            else:
                self.ie = Taskflow('information_extraction', schema=self.schema, device_id=int(device),
                                   task_path=model_path)

        self.logger.info(model_path)

    def init_review_result(self):
        return {schema: {} for schema in self.schema}

    def check_data_func(self):
        if self.device == "cpu":
            self.logger.debug(self.data)
            # exit()
            res = self.predictor.predict([self.data])
        else:
            res = self.ie(self.data)
        self.logger.debug(pformat(res))
        return res

    def rule_judge(self, extraction_res):
        self.logger.debug("res: {}".format(extraction_res))
        for index, row in self.config.iterrows():
            res_dict = {}

            if row['schema'] in extraction_res:
                if self.usr == "Part A":
                    res_dict["法律建议"] = row["A pos legal advice"]
                else:
                    res_dict["法律建议"] = row["B pos legal advice"]

                extraction_con = extraction_res[row['schema']]

                if "身份证校验" == row["pos rule"]:
                    rule_func.check_id_card(row, extraction_con, res_dict)
                elif "房屋租赁期限审核" == row["pos rule"]:
                    rule_func.check_house_application(row, extraction_con, res_dict)
                elif "借款用途审核" == row["pos rule"]:
                    rule_func.check_loan_application(row, extraction_con, res_dict)
                elif "日期内部关联" == row["pos rule"]:
                    rule_func.check_date_relation(row, extraction_con, res_dict)
                elif "日期外部关联【还款日期-借款日期】" == row["pos rule"]:
                    # self.logger.debug(extraction_res["还款日期"][0]["text"])
                    # self.logger.debug(extraction_res["借款日期"][0]["text"])
                    try:
                        rule_func.check_date_outside(row, extraction_con, res_dict,
                                                     extraction_res["借款日期"][0]["text"],
                                                     extraction_res["还款日期"][0]["text"])
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error(extraction_res)
                elif "日期外部关房屋租赁期限】" == row["pos rule"]:
                    # print(extraction_res["房屋租赁期限"][0]["text"])
                    # exit()
                    rule_func.check_hose_date_outside(row, extraction_con, res_dict,
                                                      extraction_res["房屋租赁期限"][0]["text"])
                elif "正式工资审核" == row["pos rule"]:
                    rule_func.check_wage(row, extraction_con, res_dict)
                elif "试用期工资审核" == row["pos rule"]:
                    if "劳动报酬" in self.review_result:
                        rule_func.check_probation_wage(row, extraction_con, res_dict,
                                                       self.review_result['劳动报酬']["内容"])
                    elif "工资" in self.review_result:
                        rule_func.check_probation_wage(row, extraction_con, res_dict,
                                                       self.review_result['工资']["内容"])
                    else:
                        pass
                elif "违约金审核" == row["pos rule"]:
                    rule_func.check_penalty(row, extraction_con, res_dict)
                elif "民间借贷利率审核" == row["pos rule"] or "逾期利率审核" == row["pos rule"]:
                    rule_func.check_rate(row, extraction_con, res_dict)
                elif "金额相等" == row["pos rule"]:
                    rule_func.check_amount_equal(row, extraction_con, res_dict)
                elif "竞业限制期限审核" == row["pos rule"]:
                    rule_func.check_competition_limit(row, extraction_con, res_dict)
                elif "房屋租赁期限" == row["pos rule"]:
                    rule_func.check_house_lease_term(row, extraction_con, res_dict)

                elif "违法" == row["pos rule"]:
                    res_dict["审核结果"] = "不通过"
                    res_dict["内容"] = extraction_con[0]['text']
                    res_dict["start"] = extraction_con[0]['start']
                    res_dict["end"] = extraction_con[0]['end']

                else:
                    res_dict["审核结果"] = "通过"
                    if len(extraction_con) == 1:
                        res_dict["内容"] = extraction_con[0]['text']
                        res_dict["start"] = extraction_con[0]['start']
                        res_dict["end"] = extraction_con[0]['end']
                    # 审核项目如果出现了不止一次
                    else:
                        self.logger.debug(extraction_con)
                        res_dict["内容"] = ''
                        res_dict["start"] = ''
                        res_dict["end"] = ''
                        for con in extraction_con:
                            res_dict["内容"] += con['text'] + '#'
                            res_dict["start"] = str(con['start']) + "#"
                            res_dict["end"] = str(con['end']) + '#'

            elif row['pos keywords'] != "" and len(re.findall(row['pos keywords'], self.data)) > 0:
                res_dict["审核结果"] = "通过"
                res_dict["内容"] = row['schema']

            elif row['neg rule'] == "未识别，不作审核" or row['neg rule'] == "未识别，不做审核":
                res_dict = {}

            else:
                res_dict["审核结果"] = "不通过"
                res_dict["内容"] = "没有该项目内容"
                res_dict["法律建议"] = row["neg legal advice"]

            if res_dict != {}:
                res_dict['法律依据'] = row['legal basis']
                res_dict['风险等级'] = row['risk level']
                res_dict["风险点"] = row["risk statement"]

            self.review_result[row['schema']].update(res_dict)


if __name__ == '__main__':
    import time

    contract_type = "maimai"
    acknowledgement = BasicUIEAcknowledgement(config_path="DocumentReview/Config/{}.csv".format(contract_type),
                                              log_level="INFO",
                                              # model_path="model/uie_model/new/{}/model_best/".format(contract_type),
                                              model_path="model/uie_model/export_cpu/{}/inference".format(
                                                  contract_type),
                                              device="cpu")
    print("## First Time ##")
    localtime = time.time()
    acknowledgement.review_main(content="data/DocData/{}/test.docx".format(contract_type), mode="docx", usr="Part A")
    pprint(acknowledgement.review_result, sort_dicts=False)
    print('use time: {}'.format(time.time() - localtime))

    # print("## Second Time ##")
    # acknowledgement.review_main(content="data/DocData/{}/test.docx".format(contract_type), mode="docx", usr="Part A")
    # pprint(acknowledgement.review_result, sort_dicts=False)
