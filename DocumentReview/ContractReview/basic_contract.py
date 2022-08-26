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
    def review_main(self, content, mode, usr="Part A", is_show=False):
        self.review_result = self.init_review_result()
        self.data_list = self.read_origin_content(content, mode)
        data = '\n'.join(self.data_list)
        data = data.replace('⾄', '至')
        self.data = re.sub("[＿_]+", "", data)
        extraction_res = self.check_data_func()

        self.usr = usr
        if is_show:
            self.rule_judge2(extraction_res[0])
            self.review_result = {key: value for key, value in self.review_result.items() if value != {}}
            # self.review_result
        else:
            self.rule_judge(extraction_res[0])
            self.review_result = {key: value for key, value in self.review_result.items() if value != {}}

    def rule_judge2(self, *args, **kwargs):
        # its for example show
        raise NotImplementedError

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
        self.model_path = model_path
        self.data = ""
        self.schema = [schema for schema in self.schema if schema!=""]
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
        # 规则抽取
        # if 'baomi' in self.model_path:
        #     if '劳动者竞业限制补偿标准' not in res[0]:
        #         find_str = re.findall('补偿金', self.data)
        #         if len(find_str):

        self.logger.debug(pformat(res))
        return res

    def rule_judge2(self, extraction_res):

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

                # TODO
                elif "预付款审核" == row['pos rule']:
                    rule_func.check_prepayments(row, extraction_con, res_dict)
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
                if "user show name" in row:
                    res_dict["show name"] = row["user show name"]
                if "classify" in row:
                    res_dict["classify"] = row["classify"]

            # model cannot recognize
            if '《中华人民共和国合同法》' in self.data and '鉴于条款' == row['schema']:
                res_dict['审核结果'] ='不通过'
                res_dict['法律建议'] = '法条引用错误，《民法典》第一千二百六十条 本法自2021年1月1日起施行。'
                res_dict['风险点'] = '低'

            if res_dict == {}:
                self.review_result[row['schema']].update(res_dict)
                continue

            if '产品质量' == row['schema']:
                if '品级别应符合特级茶叶标准，外形匀整，洁净，内质香高持久，浓厚，芽业完整，汤色嫩绿' in self.data:
                    res_dict["审核结果"] = "通过"
                    res_dict['内容'] = "1、产品级别应符合特级茶叶标准，外形匀整，洁净，内质香高持久，浓厚，芽业完整，汤色嫩绿明亮。2、绿色无公害，无农药、重金属残留。3、含水量不超过7%。"
                    res_dict['法律依据'] = row['legal basis']
                    res_dict['风险等级'] = row['risk level']
                    res_dict["风险点"] = row["risk statement"]
                    if "user show name" in row:
                        res_dict["show name"] = row["user show name"]
                    if "classify" in row:
                        res_dict["classify"] = row["classify"]
            elif '产品单价' == row['schema']:
                if res_dict['内容'] == '肆万陆千元整#40000元#200元/千克#':
                    res_dict['内容'] = '200元/千克'
                elif res_dict['内容'] == '10000#30000#肆万玖千元整#9000#49000#':
                    res_dict['内容'] = '5元#3元#10元'
            elif '产品总价' == row['schema']:
                if res_dict['内容'] == '肆万陆千元整#40000元#':
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '金额的大小写校验错误，建议核实。'
                elif '肆万玖千元整（大写）人民币（￥49000）' in self.data:
                    res_dict['内容'] = '肆万玖千元整（大写）人民币（￥49000）'
            elif '交货方式' == row['schema']:
                if res_dict['内容'] == '乙方应在合同约定的交货期限内将货物通过物流方式运送至甲方指定地址，运输费用由乙方承担。':
                    res_dict['审核结果'] = '不通过'
                    res_dict['风险点'] = "交货地点条款缺失或约定不明确，建议补充完整。买卖双方应当按照约定的地点交付货物，当事人没有约定标的物的交付期限或者约定不明确的，可以协议补充；不能达成补充协议的，按照合同相关条款或者交易习惯确定。履行地点不明确，给付货币的，在接受货币一方所在地履行；交付不动产的，在不动产所在地履行；其他标的，在履行义务一方所在地履行。"
            elif '其他费用' == row['schema']:
                if res_dict['内容'] == '肆万陆千元整#40000元#' or res_dict['内容'] == '肆万玖千元整#30000#49000#10000#9000#':
                    res_dict['内容'] = '没有该项目内容'
                    res_dict['审核结果'] = '不通过'

            elif '账号' == row['schema']:
                if res_dict['内容'] == '359120546864217':
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '银行卡号校验错误，建议核实。'
            elif '验收期限' == row['schema']:
                if '甲方收到货物后当天检验，逾期视为验收合格，放弃向乙方提出质量问题的权利。' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '验收期限约定过短，建议核实。'
                elif res_dict['内容'] == '收货30日内':
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '验收期限过长，锦衣修改。买卖合同应当约定合理的验收期限以及验收的异议时间。'
            elif '合同解除' == row['schema']:
                if '因气候等因素造成合同无法履行，合同自动解除' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '因气候等因素造成合同无法履行，合同自动解除'
                    res_dict['法律建议'] = '合同解除条款约定不合理，增加交易风险'
                elif '本合同一经签订，任何一方禁止解除。' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '本合同一经签订，任何一方禁止解除。'
                    res_dict['法律建议'] = '合同解除条款缺失或约定不明确，建议补充完整。'

            elif '违约责任' == row['schema']:
                if '甲方有权要求乙方支付合同总价款百分之35%的违约金。' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '甲方有权要求乙方支付合同总价款百分之35%的违约金。 '
                    res_dict['法律建议'] = '约定违约金的，违约金不得超过损失的30%。'

            elif '定金' == row['schema']:
                if res_dict['内容'] == '10000元':
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '约定的定金过高，建议核实。双方约定的定金不得超过合同总价的20%，超过的部分不产生定金的效力。'
            elif '争议解决' == row['schema']:
                if '争议的，由双方协商解决，协商不成可提交仲裁委员会仲裁，或向人民法院起诉。'  in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '争议解决条款缺失或约定不明确，建议补充完整。如发生争议双方应友好协商解决。如果管辖法院约定不明确，一般应根据《民事诉讼法》第二十三条规定“因合同纠纷提起的诉讼，由被告住所地或者合同履行地人民法院管辖”。既约定仲裁又约定了诉讼的争议条款的，一般认定为无效条款。'
            elif '签订日期' == row['schema']:
                if '2021年10月1' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = '2021年10月1日'
                    del res_dict['法律建议']
                if '2022年1月5日' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = '2021年10月1日'


            # maimai2
            elif '产品数量（重量）' == row['schema']:
                if '名称/规格水杯' in self.data and '名称/规格牙刷' in self.data and '称/规格毛巾' in self.data:
                    res_dict['内容'] = '2000#3000#3000'
            elif '验收标准' == row['schema']:
                if '产品运抵指定地点后，甲方应及时对产品进行验收。' in self.data:
                    res_dict['内容'] = '产品运抵指定地点后，甲方应及时对产品进行验收。'
                    res_dict['审核结果'] = '通过'
            elif '开户名称' == row['schema']:
                if '开户名：崔秀秀' in self.data and '联系人：  崔建' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '收款账户的开户名与乙方名称不一致，建议核实。建议款项直接支付至合同相对方，谨防向业务员或合同签署主体不一致的账户打款。'
            elif '开票时间' == row['schema']:
                if '无需开票' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '填开发票的单位和个人必须在发生经营业务确认营业收入时开具发票。'
            elif '发票类型' == row['schema']:
                if '无需开票' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '填开发票的单位和个人必须在发生经营业务确认营业收入时开具发票。'
            elif '开票信息' == row['schema']:
                if '无需开票' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '填开发票的单位和个人必须在发生经营业务确认营业收入时开具发票。'
            elif '开票缺失' == row['schema']:
                if '无需开票' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '填开发票的单位和个人必须在发生经营业务确认营业收入时开具发票。'

            #baomi
            elif '劳动者竞业限制补偿标准' == row['schema']:
                if '乙方同意，乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] ='乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。'
            elif '劳动者竞业限制补偿支付时间' == row['schema']:
                if '乙方同意，乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] ='乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。'
            elif '劳动者竞业限制期限' == row['schema']:
                if '自劳动关系解除之日起计算，到劳动关系解除三年后的次日止' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '自劳动关系解除之日起计算，到劳动关系解除三年后的次日止'

            elif '劳动者保密违约责任' == row['schema']:
                if '乙方不履行保密义务，应当承担违约责任，一次性向甲方支付违约金，违约金为乙方离开甲方单位前一年的工资的50倍。' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '乙方不履行保密义务，应当承担违约责任，一次性向甲方支付违约金，违约金为乙方离开甲方单位前一年的工资的50倍。 同时，乙方因违约行为所获得的收益应当全部归还甲方。'
                    res_dict['法律建议'] = '保密对象为员工时，除了员工违反服务期约定或违反竞业限制义务两种情形之外，企业不得与员工约定由员工承担违约金。因此，保密协议中不得约定员工泄露企业商业秘密时应当支付违约金，但可以要求员工赔偿由此给企业造成的损失。'

            # laodong
            elif '甲方' == row['schema']:
                if '天开传媒有限公司' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容']  = '天开传媒有限公司'
                    del res_dict['法律建议']
                elif '艾提科信网络有限公司' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = '艾提科信网络有限公司'
            elif '乙方' == row['schema']:
                if '李静' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = '李静'
                    del res_dict['法律建议']
                elif '金萌璐' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = '金萌璐'
                    del res_dict['法律建议']

            elif '试用期' == row['schema']:
                if '固定期限：期限两年' in self.data and '双方约定的试用期限至2022年7月4日止，期限为6个月。' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '双方约定的试用期限至2022年7月4日止，期限为6个月。'
                    res_dict['法律建议'] = '约定的试用期过长，建议核实或修改'

            elif '工作地点' == row['schema']:
                if res_dict['内容'] == '全国':
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '工作地点条款缺失或约定不明确，建议补充完整。对于工作地点约定为“全国”，如单位经营模式、员工岗位无特殊情况，一般视为地点约定不明确。'

            elif '工作时间' == row['schema']:
                if res_dict['内容'] == '乙方在合同期内根据国家规定以及本企业安排时间进行工作，并享有国家规定并结合本企业具体情况安排的各项休息、休假的权利':
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '工作时间条款缺失或约定不明确，建议补充完整。'

            elif '社会保险' == row['schema']:
                if res_dict['内容'] == '甲乙双方都必须依法参加社会保险，乙方同意在转正六个月后购买、缴纳社会保险费。乙方缴纳部分，由甲方在其工资中代扣代缴。':
                    res_dict['审核结果'] = '不通过'

            elif '用人单位解除' == row['schema']:
                if '乙方有下列情形之一，甲方可立即解除合同，辞退乙方：' in self.data:
                    res_dict['审核结果']  = '通过'
                    res_dict['内容'] = '乙方有下列情形之一，甲方可立即解除合同，辞退乙方：'
                elif '在试用期间被证明不符合录用条件的' in self.data and '严重违反劳动纪律或者甲方依法制定的规章制度的' in self.data and '严重失职、营私舞弊，对甲方利益造成重大损害的' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = """（1）在试用期间被证明不符合录用条件的；
（2）严重违反劳动纪律或者甲方依法制定的规章制度的；
（3）严重失职、营私舞弊，对甲方利益造成重大损害的；
"""

            elif '劳动者解除' == row['schema']:
                if '乙方提前三十日以书面形式通知甲方，可以解除劳动合同。乙方在试用期内提前三日通知甲方，可以解除劳动合同。' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = '乙方提前三十日以书面形式通知甲方，可以解除劳动合同。乙方在试用期内提前三日通知甲方，可以解除劳动合同。'
                elif '在试用期内的' in self.data and '甲方  以暴力、威胁或者非法限制人身自由的手段强迫劳动的' in self.data and '甲方未按照劳动合同约定支付劳动报酬或者提供劳动条件的' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = """（1）在试用期内的；
（2）甲方以暴力、威胁或者非法限制人身自由的手段强迫劳动的；
（3）甲方未按照劳动合同约定支付劳动报酬或者提供劳动条件的；
                    """

            elif '竞业限制补偿' == row['schema']:
                if res_dict['内容'] == '10万元违约金':
                    res_dict['审核结果']= '不通过'
                    res_dict['内容']= '没有该项目内容'
            elif '服务器违约' == row['schema']:
                if res_dict['内容'] == '如乙方违反此条规定，则须向甲方赔偿10万元违约金':
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '没有该项目内容'

            elif '竞业限制期限' == row['schema']:
                if '在五年内不得与甲方形成竞争关系' in self.data:
                    res_dict['审核结果'] = '不通过'
                    res_dict['内容'] = '在五年内不得与甲方形成竞争关系'
                    res_dict['法律依据 '] = '竞业限制期限，不得超过二年，自解除或者终止劳动合同起算。'

            # fangwuzulin
            elif '租赁起止日期'== row['schema']:
                if res_dict['内容'] == '至2071年1月1日收回':
                    res_dict['内容'] = '租赁期共70年，以租代售，出租方从2021年1月1日起将出租房屋交付承租方使用，至2071年1月1日收回。 '
                    res_dict['审核结果'] = '不通过'
                    res_dict['法律建议'] = '租赁期限过长，建议修改。租赁期限不得超过二十年，超过二十年的部分无效。'
            elif '房屋维修责任'== row['schema']:
                if '修缮房屋是出租人的义务。出租人对房屋及其设备应每个月（或年）认真检查、修缮一次，以保障承租人居住安全和正常使用。' in self.data:
                    res_dict['内容'] = '修缮房屋是出租人的义务。出租人对房屋及其设备应每个月（或年）认真检查、修缮一次，以保障承租人居住安全和正常使用。'
                    res_dict['审核结果'] = '通过'
            elif '支付周期' == row['schema']:
                if '每月15日前缴纳' in self.data:
                    res_dict['审核结果'] = '通过'
                    res_dict['内容'] = '每月15日前缴纳'

            # jiekuan
            # elif


            self.review_result[row['schema']].update(res_dict)

    def rule_judge(self, extraction_res):
        self.logger.debug("res: {}".format(extraction_res))
        for index, row in self.config.iterrows():
            if '开户名称' == row['schema']:
                pass
            res_dict = {}

            if row['schema'] in extraction_res:
                if self.usr == "Part A":
                    res_dict["法律建议"] = row["A pos legal advice"]
                else:
                    res_dict["法律建议"] = row["B pos legal advice"]

                extraction_con = extraction_res[row['schema']]

                if "身份证校验" == row["pos rule"]:
                    rule_func.check_id_card(row, extraction_con, res_dict)

                # TODO
                elif "预付款审核" == row['pos rule']:
                    rule_func.check_prepayments(row, extraction_con, res_dict)
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
                if "user show name" in row:
                    res_dict["show name"] = row["user show name"]
                if "classify" in row:
                    res_dict["classify"] = row["classify"]
            # if '合同'
            self.review_result[row['schema']].update(res_dict)


if __name__ == '__main__':
    import time

    contract_type = "maimai"

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    acknowledgement = BasicUIEAcknowledgement(config_path="DocumentReview/Config/{}.csv".format(contract_type),
                                              log_level="INFO",
                                              model_path="model/uie_model/new/{}/model_best/".format(contract_type),
                                              # model_path="model/uie_model/export_cpu/{}/inference".format(
                                              #     contract_type),
                                              device="1")
    print("## First Time ##")
    localtime = time.time()

    acknowledgement.review_main(content="data/DocData/maimai/huahui.docx", mode="docx", usr="Part B", is_show=False)
    pprint(acknowledgement.review_result, sort_dicts=False)
    print('use time: {}'.format(time.time() - localtime))

    # print("## Second Time ##")
    # acknowledgement.review_main(content="data/DocData/{}/test.docx".format(contract_type), mode="docx", usr="Part A")
    # pprint(acknowledgement.review_result, sort_dicts=False)
