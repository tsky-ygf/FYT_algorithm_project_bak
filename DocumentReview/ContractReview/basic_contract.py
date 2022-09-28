#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 13:31
# @Author  : Adolf
# @Site    :
# @File    : basic_contract.py
# @Software: PyCharm
import os
import re
# import uuid

import pandas as pd
from collections import OrderedDict

# from Utils import Logger
from Utils import get_logger
from DocumentReview.ParseFile.parse_word import read_docx_file

from paddlenlp import Taskflow
# from id_validator import validator

from pprint import pprint, pformat
from DocumentReview.ContractReview import rule_func
from DocumentReview.UIETool.deploy.uie_predictor import UIEPredictor

from Utils.logger import print_run_time


class BasicAcknowledgement:
    def __init__(self, config_path, log_level='INFO', logger_file=None, *args, **kwargs):
        self.logger = get_logger(level=log_level, console=True, logger_file=logger_file)
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
    def review_main(self, content, mode, usr="party_a"):
        self.review_result = self.init_review_result()
        self.data_list = self.read_origin_content(content, mode)
        data = '\n'.join(self.data_list)
        data = data.replace('⾄', '至').replace('中华⼈民', '中华人民').replace(' ', ' ').replace(u'\xa0', ' ')
        self.data = re.sub("[＿_]+", "", data)
        extraction_res = self.check_data_func()
        self.usr = usr
        self.rule_judge(extraction_res[0])
        self.review_result = {key: value for key, value in self.review_result.items() if value != {}}

        return_result = []
        self.logger.success("review_result: {}".format(pformat(self.review_result)))
        for review_point, review_result in self.review_result.items():
            return_result.append({
                "review_point": review_point,
                "show_name": review_result.get("show name") if review_result.get("show name") else review_point,
                "review_result": review_result.get("审核结果", ""),
                "review_content": review_result.get("内容", ""),
                "review_content_start": review_result.get("start", -1),
                "review_content_end": review_result.get("end", -1),
                "legal_advice": review_result.get("法律建议", ""),
                "legal_basis": review_result.get("法律依据", ""),
                "risk_level": review_result.get("风险等级", ""),
                "risk_point": review_result.get("风险点", "")
            })

        self.logger.success(return_result)
        return return_result

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
            # 数据处理统一写在文件转文字的接口中
            # content = content.replace(" ", "").replace("\u3000", "")
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
    device_id = "-1"
    schema = []


class BasicUIEAcknowledgement(BasicAcknowledgement):

    def __init__(self, model_path='', device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.schema = list(set(self.config['schema'].tolist()))
        self.device = device
        self.schema = self.config['schema'].tolist()
        # self.review_result = {schema: {} for schema in self.schema}
        self.model_path = model_path
        self.data = ""
        self.schema = [schema for schema in self.schema if schema != ""]
        if self.device == "cpu":
            args = InferArgs()
            args.model_path_prefix = model_path
            args.schema = self.schema
            self.predictor = UIEPredictor(args)
            # self.ie = Taskflow('information_extraction', schema=self.schema, device_id=-1, task_path=model_path)
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
            # res = self.ie(self.data)
        else:
            res = self.ie(self.data)

        self.logger.debug(pformat(res))
        return res

    def rule_judge(self, extraction_res):
        self.logger.debug("res: {}".format(extraction_res))
        for index, row in self.config.iterrows():
            res_dict = {}

            if row['schema'] in extraction_res:
                extraction_con = extraction_res[row['schema']]
                if self.usr == "party_a":
                    res_dict["法律建议"] = row["A pos legal advice"]
                else:
                    res_dict["法律建议"] = row["B pos legal advice"]
                if "身份证校验" == row["pos rule"]:
                    rule_func.check_id_card(row, extraction_con, res_dict)

                elif '一次性付款条款审核' == row['pos rule']:
                    rule_func.check_once_pay(row, extraction_con, res_dict)
                elif '房屋租赁押金审核' in row['pos rule']:
                    rule_func.check_deposit(row, extraction_res, res_dict)
                # TODO
                # elif "预付款审核" == row['pos rule']:
                #     rule_func.check_prepayments(row, extraction_con, res_dict)
                elif '产品名称审核' == row['pos rule']:
                    rule_func.check_product_name(row, extraction_con, res_dict)
                elif '试用期期限审核' == row['pos rule']:
                    rule_func.check_trial_period(row, extraction_con, res_dict)
                elif '劳务合同争议解决方式审核' == row['pos rule']:
                    rule_func.labor_contract_dispute_resolution(row, extraction_con, res_dict)
                elif '竞业限制补偿标准审核' == row['pos rule']:
                    rule_func.compensation_standard_for_non_compete(row, extraction_con, res_dict)
                #
                elif '竞业限制补偿支付时间审核' == row['pos rule']:
                    rule_func.check_noncompete_compensation_payment_time(row, extraction_con, res_dict)
                #
                elif '支付周期审核' == row['pos rule']:
                    rule_func.check_housing_lease_payment_cycle(row, extraction_con, res_dict)
                # model cannot recognize but implemented in pos keywords
                # elif '房屋租赁合同管辖法院审核' == row['pos rule']:
                #     rule_func.check_housing_tenancy_court(row, extraction_con, res_dict)
                # TODO
                elif "违约金审核" == row["pos rule"]:
                    rule_func.check_penalty(row, extraction_con, res_dict)

                elif "房屋租赁期限审核" == row["pos rule"]:
                    rule_func.check_house_lease_term(row, extraction_con, res_dict)
                elif "借款用途审核" == row["pos rule"]:
                    rule_func.check_loan_application(row, extraction_con, res_dict)
                elif "日期内部关联" == row["pos rule"]:
                    rule_func.check_date_relation(row, extraction_con, res_dict)
                elif "日期外部关联【还款日期-借款日期】" == row["pos rule"]:
                    try:
                        if not '借款日期' in extraction_res or '还款日期' not in extraction_res:
                            res_dict['审核结果'] = '不通过'
                            res_dict['法律建议'] = row['jiaoyan error advice']
                        else:
                            rule_func.check_date_outside(row, extraction_con, res_dict,
                                                         extraction_res["借款日期"][0]["text"],
                                                         extraction_res["还款日期"][0]["text"])
                    except Exception as e:
                        self.logger.error('-' * 50 + 'error!')
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
                    res_dict['风险等级'] = row['risk level']
                    res_dict["风险点"] = row["risk statement"]

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

                # model cannot recognize
                if ('甲方' == row['schema'] or '甲方联系方式' == row['schema'] or '甲方地址' == row['schema']
                    or '甲方身份证号/统一社会信用代码' == row['schema']) and len(
                    extraction_con) > 1:
                    res_dict["内容"] = extraction_con[0]['text']
                    res_dict["start"] = str(extraction_con[0]['start'])
                    res_dict["end"] = str(extraction_con[0]['end'])
                elif ('乙方' == row['schema'] or '乙方联系方式' == row['schema'] or '乙方地址' == row[
                    'schema'] or '乙方身份证号/统一社会信用代码' == row['schema']) and len(extraction_con) > 1:
                    res_dict["内容"] = extraction_con[1]['text']
                    res_dict["start"] = str(extraction_con[1]['start'])
                    res_dict["end"] = str(extraction_con[1]['end'])

            elif row['pos keywords'] != "" and len(re.findall(row['pos keywords'], self.data)) > 0:
                res_dict["审核结果"] = "通过"
                r = re.findall(row['pos keywords'], self.data)
                if isinstance(r[0], str):
                    r = '，'.join(r)
                    res_dict["内容"] = r
                else:
                    res_dict["内容"] = row['schema']
                if r in self.data:
                    self.add_start_end(r, res_dict)

            elif row['neg rule'] == "未识别，不作审核" or row['neg rule'] == "未识别，不做审核":
                res_dict = {}
            else:
                res_dict["审核结果"] = "不通过"
                res_dict["内容"] = "没有该项目内容"
                res_dict["法律建议"] = row["neg legal advice"]

            # model cannot recognize
            r_temp = re.findall(
                r'《中华人民共和国婚姻法》|《中华人民共和国继承法》|《中华人民共和国民法通则》|《中华人民共和国收养法》|《中华人民共和国担保法》|《中华人民共和国合同法》|《中华人民共和国物权法》|《中华人民共和国侵权责任法》|《中华人民共和国民法总则》',
                self.data)
            if '鉴于条款' == row['schema'] and len(r_temp) > 0:
                res_dict['审核结果'] = '不通过'
                res_dict[
                    '法律建议'] = '法条引用错误，《民法典》第一千二百六十条 本法自2021年1月1日起施行。《中华人民共和国婚姻法》、《中华人民共和国继承法》、《中华人民共和国民法通则》、《中华人民共和国收养法》、《中华人民共和国担保法》、《中华人民共和国合同法》、《中华人民共和国物权法》、《中华人民共和国侵权责任法》、《中华人民共和国民法总则》同时废止。'
                res_dict['风险点'] = '低'
                if '内容' not in res_dict:
                    res_dict['内容'] = r_temp[0]

            if res_dict != {}:
                res_dict['法律依据'] = row['legal basis']
                res_dict['风险等级'] = row['risk level']
                res_dict["风险点"] = row["risk statement"]
                if "user show name" in row:
                    res_dict["show name"] = row["user show name"]
                if "classify" in row:
                    res_dict["classify"] = row["classify"]

            self.review_result[row['schema']].update(res_dict)

    def add_start_end(self, content, res_dict):
        if content in self.data:
            index_r = self.data.index(content)
            start_t = index_r
            end_t = index_r + len(content)
            res_dict['start'] = start_t
            res_dict['end'] = end_t


if __name__ == '__main__':
    import time

    contract_type = "fangwuzulin"

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    acknowledgement = BasicUIEAcknowledgement(config_path="DocumentReview/Config/{}.csv".format(contract_type),
                                              log_level="INFO",
                                              model_path="model/uie_model/new/{}/model_best/".format(contract_type),
                                              # model_path="model/uie_model/export_cpu/{}/inference".format(
                                              #     contract_type),
                                              device="1")
    print("## First Time ##")
    localtime = time.time()

    acknowledgement.review_main(content="data/DocData/fangwuzulin/fwzl1.docx", mode="docx", usr="Part B")
    pprint(acknowledgement.review_result, sort_dicts=False)
    print('use time: {}'.format(time.time() - localtime))

    # print("## Second Time ##")
    # acknowledgement.review_main(content="data/DocData/{}/test.docx".format(contract_type), mode="docx", usr="Part A")
    # pprint(acknowledgement.review_result, sort_dicts=False)
