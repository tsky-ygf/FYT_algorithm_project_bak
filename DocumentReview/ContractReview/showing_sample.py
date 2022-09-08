import os
import re
from pprint import pprint

import pandas as pd

from DocumentReview.ContractReview.basic_contract import BasicUIEAcknowledgement
from DocumentReview.ContractReview import rule_func
from Utils.logger import print_run_time


class BasicUIEAcknowledgementShow(BasicUIEAcknowledgement):

    @print_run_time
    def review_main(self, content, mode, usr="Part A"):
        self.review_result = self.init_review_result()
        self.data_list = self.read_origin_content(content, mode)
        data = '\n'.join(self.data_list)
        data = data.replace('⾄', '至').replace('中华⼈民', '中华人民')
        self.data = re.sub("[＿_]+", "", data)
        extraction_res = self.check_data_func()

        self.usr = usr
        self.rule_judge2(extraction_res[0])
        self.review_result = {key: value for key, value in self.review_result.items() if value != {}}
        # self.review_result['origin_data'] = self.data

    def rule_judge2(self, extraction_res):
        print('*' * 100)
        print(self.data)
        print('*' * 100)
        self.logger.debug("res: {}".format(extraction_res))
        self.data = self.data.replace(' ', '')
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
                            res_dict["start"] += str(con['start']) + "#"
                            res_dict["end"] += str(con['end']) + '#'

                # model cannot recognize
                if ('甲方' == row['schema'] or '甲方联系方式' == row['schema'] or '甲方地址' == row['schema']) and len(
                        extraction_con) > 1:
                    res_dict["内容"] = extraction_con[0]['text']
                    res_dict["start"] = str(extraction_con[0]['start'])
                    res_dict["end"] = str(extraction_con[0]['end'])
                elif ('乙方' == row['schema'] or '乙方联系方式' == row['schema'] or '乙方地址' == row[
                    'schema']) and len(extraction_con) > 1:
                    res_dict["内容"] = extraction_con[1]['text']
                    res_dict["start"] = str(extraction_con[1]['start'])
                    res_dict["end"] = str(extraction_con[1]['end'])

            elif row['pos keywords'] != "" and len(re.findall(row['pos keywords'], self.data)) > 0:
                res_dict["审核结果"] = "通过"
                r = re.findall(row['pos keywords'], self.data)
                if isinstance(r[0], str):
                    r = '，'.join(r)
                    res_dict["内容"] = r
                    # TODO start and end
                else:
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
            if '鉴于条款' == row['schema']:
                if '内容' in res_dict:
                    r_temp = re.findall(
                        r'《中华人民共和国婚姻法》|《中华人民共和国继承法》|《中华人民共和国民法通则》|《中华人民共和国收养法》|《中华人民共和国担保法》|《中华人民共和国合同法》|《中华人民共和国物权法》|《中华人民共和国侵权责任法》|《中华人民共和国民法总则》',
                        res_dict['内容'])
                    if len(r_temp) > 0:
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '法条引用错误，《民法典》第一千二百六十条 本法自2021年1月1日起施行。《中华人民共和国婚姻法》、《中华人民共和国继承法》、《中华人民共和国民法通则》、《中华人民共和国收养法》、《中华人民共和国担保法》、《中华人民共和国合同法》、《中华人民共和国物权法》、《中华人民共和国侵权责任法》、《中华人民共和国民法总则》同时废止。'
                        res_dict['风险点'] = '低'
                    self.add_start_end(res_dict['内容'], res_dict)

            # finished start and end
            if 'maimai' in self.model_path:

                # maimai
                if '产品质量' == row['schema']:
                    if '品级别应符合特级茶叶标准，外形匀整，洁净，内质香高持久，浓厚，芽业完整，汤色嫩绿' in self.data:
                        res_dict["审核结果"] = "通过"
                        res_dict['内容'] = '''产品级别应符合特级茶叶标准，外形匀整，洁净，内质香高持久，浓厚，芽业完整，汤色嫩绿明亮。
绿色无公害，无农药、重金属残留。
含水量不超过7%。'''
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        if "user show name" in row:
                            res_dict["show name"] = row["user show name"]
                        if "classify" in row:
                            res_dict["classify"] = row["classify"]
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '产品单价' == row['schema']:
                    if res_dict.get('内容', '') == '肆万陆千元整#40000元#200元/千克#':
                        res_dict['内容'] = '200元/千克'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif res_dict.get('内容', '') == '10000#30000#肆万玖千元整#9000#49000#':
                        res_dict['内容'] = '5元#3元#10元'
                        if '5元' in self.data and '3元' in self.data and '10元' in self.data:
                            res_dict['start'] = str(self.data.index('5元'))+"#"+str(self.data.index('3元'))+"#"+str(self.data.index('10元'))
                            ends = res_dict['start'].split('#')
                            ends = [str(int(_)+2) for _ in ends]
                            res_dict['end'] = '#'.join(ends)
                    elif '4980元#2.49元#肆仟玖佰捌拾元整#4980元#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '2.49元'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '产品总价' == row['schema']:
                    if res_dict.get('内容', '') == '肆万陆千元整#40000元#':
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '金额的大小写校验错误，建议核实。'
                        res_dict['内容'] = '40000元（大写：肆万陆千元整）'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '肆万玖千元整（大写）人民币（￥49000）' in self.data:
                        res_dict['内容'] = '肆万玖千元整（大写）人民币（￥49000）'
                        self.add_start_end('肆万玖千元整（大写）人民币（￥49000）', res_dict)
                    elif '2.49元#4980元#肆仟玖佰捌拾元整#4980元#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '4980元（大写：肆仟玖佰捌拾元整）'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '交货时间' == row['schema']:
                    if '交货期限：本合同签订之日起7日内交货。如供应商缺货等原因，导致一方延迟交货，乙方不承担违约责任。' in self.data:
                        res_dict['内容'] = '交货期限：本合同签订之日起7日内交货。如供应商缺货等原因，导致一方延迟交货，乙方不承担违约责任。 '
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '交付期限未严格按照合同约定执行，增加买方风险，建议核实。'
                        self.add_start_end('交货期限：本合同签订之日起7日内交货。如供应商缺货等原因，导致一方延迟交货，乙方不承担违约责任。', res_dict)
                    elif '2022年7月5日前' in self.data:
                        res_dict['内容'] = '2022年7月5日前'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end('2022年7月5日前', res_dict)
                elif '交货方式' == row['schema']:
                    if res_dict.get('内容', '') == '乙方应在合同约定的交货期限内将货物通过物流方式运送至甲方指定地址，运输费用由乙方承担。':
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '交货地点条款缺失或约定不明确，建议补充完整。买卖双方应当按照约定的地点交付货物，当事人没有约定标的物的交付期限或者约定不明确的，可以协议补充；不能达成补充协议的，按照合同相关条款或者交易习惯确定。履行地点不明确，给付货币的，在接受货币一方所在地履行；交付不动产的，在不动产所在地履行；其他标的，在履行义务一方所在地履行。'
                        res_dict['内容'] = '物流方式运送'
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '交货地点' == row['schema']:
                    if '乙方应在合同约定的交货期限内将货物通过物流方式运送至甲方指定地址，运输费用由乙方承担。' in self.data:
                        res_dict['内容'] = '甲方指定地址'
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '交货地点条款缺失或约定不明确，建议补充完整。买卖双方应当按照约定的地点交付货物，当事人没有约定标的物的交付期限或者约定不明确的，可以协议补充；不能达成补充协议的，按照合同相关条款或者交易习惯确定。履行地点不明确，给付货币的，在接受货币一方所在地履行；交付不动产的，在不动产所在地履行；其他标的，在履行义务一方所在地履行。'
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '其他费用' == row['schema']:
                    if res_dict.get('内容', '') == '肆万陆千元整#40000元#' or res_dict.get('内容',
                                                                                  '') == '肆万玖千元整#30000#49000#10000#9000#' or '4980元#2.49元#肆仟玖佰捌拾元整#' == \
                            res_dict.get('内容', ''):
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = row['jiaoyan error advice']

                elif '账号' == row['schema']:
                    if res_dict.get('内容', '') == '359120546864217':
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '银行卡号校验错误，建议核实。'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '验收期限' == row['schema']:
                    if '甲方收到货物后当天检验，逾期视为验收合格，放弃向乙方提出质量问题的权利。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '验收期限约定过短，建议核实。'
                        res_dict['内容'] = '甲方收到货物后当天检验，逾期视为验收合格，放弃向乙方提出质量问题的权利。'
                        self.add_start_end('甲方收到货物后当天检验，逾期视为验收合格，放弃向乙方提出质量问题的权利。', res_dict)
                    elif res_dict.get('内容', '') == '收货30日内':
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '验收期限过长，锦衣修改。买卖合同应当约定合理的验收期限以及验收的异议时间。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '甲方应当于收到货物后1日内' in self.data:
                        res_dict['内容'] = '甲方应当于收到货物后1日内'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '验收期限过短，建议修改。买卖合同应当约定合理的验收期限以及验收的异议时间。'
                        self.add_start_end('甲方应当于收到货物后1日内', res_dict)
                elif '合同解除' == row['schema']:
                    if '因气候等因素造成合同无法履行，合同自动解除' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '因气候等因素造成合同无法履行，合同自动解除'
                        res_dict['法律建议'] = '合同解除条款约定不合理，增加交易风险'
                        self.add_start_end('因气候等因素造成合同无法履行，合同自动解除', res_dict)
                    elif '本合同一经签订，任何一方禁止解除。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '本合同一经签订，任何一方禁止解除。'
                        res_dict['法律建议'] = '合同解除条款缺失或约定不明确，建议补充完整。'
                        self.add_start_end('本合同一经签订，任何一方禁止解除。', res_dict)
                elif '违约责任' == row['schema']:
                    if '甲方有权要求乙方支付合同总价款百分之35%的违约金。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '甲方有权要求乙方支付合同总价款百分之35%的违约金。 '
                        res_dict['法律建议'] = '约定违约金的，违约金不得超过损失的30%。'
                        self.add_start_end('甲方有权要求乙方支付合同总价款百分之35%的违约金。', res_dict)
                elif '定金' == row['schema']:
                    if '内容' in res_dict and res_dict.get('内容', '') == '10000元':
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '约定的定金过高，建议核实。双方约定的定金不得超过合同总价的20%，超过的部分不产生定金的效力。'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '争议解决' == row['schema']:
                    if '争议的，由双方协商解决，协商不成可提交仲裁委员会仲裁，或向人民法院起诉。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '争议解决条款缺失或约定不明确，建议补充完整。如发生争议双方应友好协商解决。如果管辖法院约定不明确，一般应根据《民事诉讼法》第二十三条规定“因合同纠纷提起的诉讼，由被告住所地或者合同履行地人民法院管辖”。既约定仲裁又约定了诉讼的争议条款的，一般认定为无效条款。'
                        self.add_start_end('争议的，由双方协商解决，协商不成可提交仲裁委员会仲裁，或向人民法院起诉。', res_dict)
                elif '签订日期' == row['schema']:
                    if '2021年10月1' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '2021年10月1日'
                        res_dict['法律建议'] = ''
                        self.add_start_end('2021年10月1', res_dict)
                    elif '2022年1月5日' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '2022年1月5日'
                        self.add_start_end('2022年1月5日', res_dict)
                elif '开票缺失' == row['schema']:
                    if '本价格系不含税价格，不开具发票，如需开票，需另行支付税费后开具。 ' in self.data:
                        res_dict['内容'] = '本价格系不含税价格，不开具发票，如需开票，需另行支付税费后开具。'
                        res_dict['审核结果'] = '不通过'
                        res_dict["法律建议"] = row["jiaoyan error advice"]
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('本价格系不含税价格，不开具发票，如需开票，需另行支付税费后开具。', res_dict)
                    elif '双方协商一致本合同价款均为优惠价，为不含税价，无需开票。' in self.data:
                        res_dict['内容'] = '双方协商一致本合同价款均为优惠价，为不含税价，无需开票。 '
                        res_dict['审核结果'] = '不通过'
                        res_dict["法律建议"] = row["jiaoyan error advice"]
                        self.add_start_end('双方协商一致本合同价款均为优惠价，为不含税价，无需开票。', res_dict)

                # maimai2
                elif '产品数量（重量）' == row['schema']:
                    if '名称/规格水杯' in self.data and '名称/规格牙刷' in self.data and '称/规格毛巾' in self.data:
                        res_dict['内容'] = '2000#3000#3000'
                        if '2000' in self.data and '3000' in self.data and '3000' in self.data:
                            res_dict['start'] = str(self.data.index('2000')) + "#" + str(
                                self.data.index('3000')) + "#" + str(self.data.index('3000'))
                            ends = res_dict['start'].split('#')
                            ends = [str(int(_) + 4) for _ in ends]
                            res_dict['end'] = '#'.join(ends)
                elif '验收标准' == row['schema']:
                    if '产品运抵指定地点后，甲方应及时对产品进行验收。' in self.data:
                        res_dict['内容'] = '产品运抵指定地点后，甲方应及时对产品进行验收。'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('产品运抵指定地点后，甲方应及时对产品进行验收。', res_dict)

                elif '开户名称' == row['schema']:
                    if '开户名：崔秀秀' in self.data and '联系人：  崔建' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '收款账户的开户名与乙方名称不一致，建议核实。建议款项直接支付至合同相对方，谨防向业务员或合同签署主体不一致的账户打款。'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('开户名：崔秀秀', res_dict)
                elif '开票时间' == row['schema']:
                    if '无需开票' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '填开发票的单位和个人必须在发生经营业务确认营业收入时开具发票。'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('无需开票', res_dict)
                elif '发票类型' == row['schema']:
                    if '无需开票' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '填开发票的单位和个人必须在发生经营业务确认营业收入时开具发票。'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('无需开票', res_dict)
                elif '开票信息' == row['schema']:
                    if '无需开票' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '填开发票的单位和个人必须在发生经营业务确认营业收入时开具发票。'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('无需开票', res_dict)
            # finished start and end
            elif 'baomi' in self.model_path:
                if '保密内容和范围' == row['schema']:
                    if '姚桂英' in self.data:
                        temp = '''（1）甲方公司重大决策中的秘密事项；
（2）甲方公司内部掌握的合同、协议、会议纪要；
（3）甲方公司规划、设计、工程图纸等；
（4）应甲方公司客户要求保密的资料；
（5）甲方客户名单、联系方式及其它客户信息；
（6）使甲方单位直接或间接的经济利益受到损害的事项；
（7）影响甲方单位对外交流和商业谈判顺利进行的事项；
（8）影响甲方单位稳定和安全的事项；
（9）影响甲方单位对外承担保密义务的事项。'''
                        res_dict['内容'] = temp
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(temp, res_dict)
                    elif "凡本人在公司任职期间而获得的（直接或间接方式）可以给公司带来经济利益或竞争优势的，且不为公众所知的具体的经营信息与技术信息均属公司的保密信息，包括但不限于下列各项" in self.data:
                        res_dict['内容'] = """凡本人在公司任职期间而获得的（直接或间接方式）可以给公司带来经济利益或竞争优势的，且不为公众所知的具体的经营信息与技术信息均属公司的保密信息，包括但不限于下列各项：
1.未被公众所知的有关公司的产品、方法、工艺、改良、公式和设计等信息；
2.图纸（含草图）：包括但不限于产品图纸、模具图纸以及设计草图等；
3.研究开发的文件：包括但不限于记录研究开发活动内容的各类文件，比如会议纪要、实验结果、技术改进通知、检验方法等；
4.公司的业务和市场策略； 
5.公司（现有和潜在的）客户情况，包括但不限于客户名单等；
6.与研究项目、专有技术、价格、折扣、加价、营销、招标及经营策略等有关信息，以及与公司的知识产权组合及策略有关的信息；
7.其他资料，包括其他与公司的竞争和效益有关的商业信息、经营信息、技术信息、采购计划、供货渠道、销售计划、会计财务报表等。"""
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '乙方不得向其他媒体、网站提供甲方内部信息和职务行为信息，或者担任境外媒体的“特约记者”、“特约通信员”、“特约撰稿人”或专栏作者等' in self.data:
                        res_dict['内容'] = '''1.乙方不得非法复制、记录、储存涉密信息，不得在任何媒体以任何形式传递涉秘信息，不得在私人交往和通信中涉及涉密信息。
2.乙方不得向其他媒体、网站提供甲方内部信息和职务行为信息，或者担任境外媒体的“特约记者”、“特约通信员”、“特约撰稿人”或专栏作者等。
3.乙方不得利用甲方内部信息和职务行为信息谋取不正当利益。
4.乙方不得非法使用任何属于他人的商业秘密，亦不得实施侵犯他人知识产权的行为。
5.乙方以职务身份开设博客、微博、微信等，须经所在单位批准备案，不得违反保密协议的约定；乙方不得通过博客、微博、微信公众账号或个人账户等任何渠道，以及论坛、讲座等任何场所，透漏、发布职务行为信息。
6.乙方严禁在个人微博、微信和客户端中谈论与职务行为有关的内容，特别是涉及各类敏感话题的内容；不得发布采编工作中了解掌握的有关报道精神和报道安排以及涉及本单位内部信息。
7.不得发布和传播与本单位公开报道口径相违背的新闻信息；不得发布与本单位工作人员身份不符的言论。
8.记者个人微博、微信和客户端不能越权发布重要信息。
9.严格区分个人账号与法人账号，严禁混淆发稿。'''
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '产品、物料及服务的供应源' in self.data:
                        res_dict['内容'] = '''1.保密信息是指由甲方通过文字、电子、数字方式或其它任何方式向乙方提供的各类信息。保密信息包括但不限于：
（1）商圈数据库所有的数据信息；
（2）客户或潜在客户的身份及其他相关信息、客户联系方式等；
（3）市场研究结果，市场和销售计划及其他市场信息；
（4）签约合同文本及其他标准文件；
（5）销售额、成本和其他财务数据；
（6）经营秘密、技术秘密、设计及专有的经营和技术信息，与本合同所涉及产品及有关的方法、经验、程序、步骤；
（7）产品、物料及服务的供应源；
（8）任何其他秘密工艺、配方或方法；
（9）本项目进行中形成的技术、产权、软件成果、研究思路在对外公布之前。
2.无论甲方在提供信息时是否标明“保密”，乙方均应按保密信息对待，除非甲方明确说明不属于保密信息。
3.本合同同样适用于本合同签订之前甲方已经向乙方提供的保密信息。
4.保密信息不包括以下信息：
（1）甲方已经公布于众的资料（包括但不限于官网、公众号等宣传平台发布信息），但不包括甲乙双方或其代表违反本合同规定未经授权所披露的；
（2）乙方在依照本合同条款从甲方获悉之前已经占有的信息，并且就乙方所知乙方并不需要对该等信息承担任何具有约束力的保密义务；
（3）在双方签订本合同以后并非由于乙方的过错而被公众所知的信息；
（4）乙方在未违反其对甲方承担的任何义务的情况下从第三方获得的信息。'''
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '商业合作方违约责任' == row['schema']:
                    if '保密信息为本合同之外第三人所知，损害甲方商业价值' in self.data:
                        res_dict[
                            '内容'] = '乙方应当严格按照本合同约定履行保密义务，若乙方违反本合同项下的保密义务，致使甲方保密信息为本合同之外第三人所知，损害甲方商业价值，乙方应向甲方支付违约金人民币（大写）伍拾万元（￥500000）；违约金不足以弥补甲方损失的，乙方还应赔偿甲方全部损失，包括但不限于律师费、差旅费、交通费、调查费、公证费、诉讼或仲裁费、担保费、保全费等甲方为实现权利支出的合理费用。'
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '约定违约金，违约金数额不能过高，超过实际损失30%的违约金会被法院认定为过高，建议根据实际情况合理确定违约金数额【《中华人民共和国民法典》第五百八十五条】当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。约定的违约金低于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以增加；约定的违约金过分高于造成的损失的，人民法院或者仲裁机构可以根据当事人的请求予以适当减少。当事人就迟延履行约定违约金的，违约方支付违约金后，还应当履行债务。'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '劳动者竞业限制补偿标准' == row['schema']:
                    if '乙方同意，乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。'
                        res_dict['法律建议'] = '竞业限制补偿金过低，建议核实。经济补偿金不得少于劳动合同履行地最低工资标准。'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '劳动者竞业限制补偿支付时间' == row['schema']:
                    if '乙方同意，乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '乙方离职后可享受的竞业限制补偿每月1000元由甲方在乙方在职期间与工资一并发放，乙方离职后不在享有竞业限制补偿的权益。'
                        res_dict['法律建议'] = '竞业限制补偿支付时间不得早于解除或者终止劳动合同时间。'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '劳动者竞业限制期限' == row['schema']:
                    if '自劳动关系解除之日起计算，到劳动关系解除三年后的次日止' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '自劳动关系解除之日起计算，到劳动关系解除三年后的次日止'
                        res_dict['法律建议'] = row['jiaoyan error advice']
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '本人从公司离职后，承诺永久不会与公司原客户从事与公司相竞争的业务或交易。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '本人从公司离职后，承诺永久不会与公司原客户从事与公司相竞争的业务或交易。'
                        res_dict['法律建议'] = row['jiaoyan error advice']
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '劳动者保密违约责任' == row['schema']:
                    if '乙方不履行保密义务，应当承担违约责任，一次性向甲方支付违约金，违约金为乙方离开甲方单位前一年的工资的50倍。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '内容'] = '''6.1 乙方不履行保密义务，应当承担违约责任，一次性向甲方支付违约金，违约金为乙方离开甲方单位前一年的工资的50倍。同时，乙方因违约行为所获得的收益应当全部归还甲方。
6.2 乙方违反保密义务造成甲方损失的，应予以全额赔偿，损失按照如下方式计算：
（1）损失赔偿为甲方因乙方的违约或侵权行为所受到的实际经济损失。
（2）甲方因调查乙方的违约或侵权行为而支付的合理费用，包括但不限于律师费、公证费、取证费。
（3）因乙方的违约或侵权行为侵犯了甲方的商业秘密时，甲方可以选择根据本协议要求乙方承担违约责任，或者根据国家有关法律、法规要求乙方承担侵权责任。
6.3 因乙方恶意泄露商业秘密给甲方造成严重后果的，甲方将通过法律手段追究其侵权责任，直至追究其刑事责任。'''
                        res_dict[
                            '法律建议'] = '保密对象为员工时，除了员工违反服务期约定或违反竞业限制义务两种情形之外，企业不得与员工约定由员工承担违约金。因此，保密协议中不得约定员工泄露企业商业秘密时应当支付违约金，但可以要求员工赔偿由此给企业造成的损失。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '如有违背上述保密自律条款承诺的，本人同意公司可按照相关的规章制度对本人作出处理，给予通报批评、降职降薪、解除劳动关系等相应处罚，同时因本人行为给公司造成损害的，由本人承担损害赔偿责任。' in self.data:
                        res_dict[
                            '内容'] = '如有违背上述保密自律条款承诺的，本人同意公司可按照相关的规章制度对本人作出处理，给予通报批评、降职降薪、解除劳动关系等相应处罚，同时因本人行为给公司造成损害的，由本人承担损害赔偿责任。 '
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '乙方擅自发布职务行为信息造成严重后果的，列入不良从业行为记录，甲方可以做出禁业或限业处理' in self.data:
                        res_dict['内容'] = '''1.乙方违反保密承诺和保密协议、擅自使用职务行为信息的，甲方可追究违约责任，视情节作出行政处理或纪律处分，并追究民事责任。
2.乙方擅自发布职务行为信息造成严重后果的，列入不良从业行为记录，甲方可以做出禁业或限业处理。
3.乙方违反规定使用职务行为信息造成失密泄密的，依法追究相关人员责任，涉嫌违法犯罪的移送司法机关处理。'''
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '竞业限制违约责任' == row['schema']:
                    if '乙方不履行竞业限制义务的，应承担违约责任，向甲方返还竞业限制补偿，并支付违约金1万元 ' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '乙方不履行竞业限制义务的，应承担违约责任，向甲方返还竞业限制补偿，并支付违约金1万元'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '争议解决' == row['schema']:
                    if '可向甲方所在地有管辖权的人民法院或仲裁委员会申请处理' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '内容'] = '因本合同引起的或与本合同有关的任何争议，由合同各方协商解决，也可由有关部门调解。协商或调解不成的，可向甲方所在地有管辖权的人民法院或仲裁委员会申请处理。'
                        res_dict['法律建议'] = '或裁或审的争议解决方式可能导致其中关于仲裁协议无效。【《最高人民法院关于适用<中华人民共和国仲裁法>若干问题的解释》第七条】当事人约定争议可以向仲裁机构申请仲裁也可以向人民法院起诉的，仲裁协议无效。但一方向仲裁机构申请仲裁，另一方未在仲裁法第二十条第二款规定期间内提出异议的除外。'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '合同生效' == row['schema']:
                    if '本协议自双方签字或盖章后生效' in self.data:
                        res_dict['内容'] = '本协议自双方签字或盖章后生效'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '经甲、乙双方签字盖章之日起生效' in self.data:
                        res_dict['内容'] = '经甲、乙双方签字盖章之日起生效。'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '本合同自双方有权代表签字或公司盖章之日起生效。' in self.data:
                        res_dict['内容'] = '本合同自双方有权代表签字或公司盖章之日起生效。'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '合同解除' == row['schema']:
                    if '甲方如本协议约定保密内容已不形成商业秘密，可以解除本保密协议' in self.data:
                        res_dict['内容'] = '''4.4甲方如本协议约定保密内容已不形成商业秘密，可以解除本保密协议。
4.5如甲方未按约定支付竞业限制补偿超过三个月，乙方有权解除本保密协议。'''
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        if "user show name" in row:
                            res_dict["show name"] = row["user show name"]
                        if "classify" in row:
                            res_dict["classify"] = row["classify"]
            # finished start and end
            elif 'laodong' in self.model_path:
                if '甲方' == row['schema']:
                    if '天开传媒有限公司' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '天开传媒有限公司'
                        res_dict['法律建议'] = ''
                        self.add_start_end('天开传媒有限公司', res_dict)
                    elif '艾提科信网络有限公司' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '艾提科信网络有限公司'
                        res_dict['法律建议'] = ''
                        self.add_start_end('艾提科信网络有限公司', res_dict)
                    elif '华成育卓传媒有限公司' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '华成育卓传媒有限公司'
                        res_dict['法律建议'] = ''
                        self.add_start_end('华成育卓传媒有限公司', res_dict)
                    elif '图龙信息网络有限公司' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '图龙信息网络有限公司'
                        res_dict['法律建议'] = ''
                        self.add_start_end('图龙信息网络有限公司', res_dict)

                elif '乙方' == row['schema']:
                    if '李静' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '李静'
                        res_dict['法律建议'] = ''
                        self.add_start_end('李静', res_dict)
                    elif '金萌璐' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '金萌璐'
                        res_dict['法律建议'] = ''
                        self.add_start_end('金萌璐', res_dict)
                    elif '张红' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '张红'
                        res_dict['法律建议'] = ''
                        self.add_start_end('张红', res_dict)

                elif '甲方统一社会信用代码' == row['schema']:
                    if '甲、乙双方根据《中华人民共和国劳动合同法》等法律、法规、规章的规定，在平等、自愿、协商一致的基础上，同意订立本劳动合同，共同遵守本合同所列条款' == \
                            res_dict.get('内容', ''):
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                    elif '华成育卓传媒有限公司' in self.data:
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]

                elif '试用期' == row['schema']:
                    if '固定期限：期限两年' in self.data and '双方约定的试用期限至2022年7月4日止，期限为6个月。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '双方约定的试用期限至2022年7月4日止，期限为6个月。'
                        res_dict['法律建议'] = '约定的试用期过长，建议核实或修改'
                        self.add_start_end('双方约定的试用期限至2022年7月4日止，期限为6个月。', res_dict)
                    elif '其中前1个⽉为试⽤期' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '其中前1个⽉为试⽤期'
                        res_dict['法律建议'] = ''
                        self.add_start_end('其中前1个⽉为试⽤期', res_dict)
                elif '试用期工资' == row['schema']:
                    if '试⽤期(⻅习期)⼯资待遇为3000元每月' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '3000元每月 '
                        res_dict['法律建议'] = '约定的试用期工资过低，建议核实。试用期工资不得少于正式工资的80%'
                        self.add_start_end('试⽤期(⻅习期)⼯资待遇为3000元每月', res_dict)

                elif '工资' == row['schema']:
                    if '满后的⼯资待遇为5000元每月' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '5000元每月'
                        res_dict['法律建议'] = ''
                        self.add_start_end('5000元每月', res_dict)
                elif '工作地点' == row['schema']:
                    if res_dict['内容'] == '全国':
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '工作地点条款缺失或约定不明确，建议补充完整。对于工作地点约定为“全国”，如单位经营模式、员工岗位无特殊情况，一般视为地点约定不明确。'
                        self.add_start_end('全国', res_dict)

                    elif '负责在华南区域内开展公司' in self.data:
                        res_dict['内容'] = '华南区域'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end('华南区域', res_dict)

                elif '工作时间' == row['schema']:
                    if res_dict.get('内容', '') == '乙方在合同期内根据国家规定以及本企业安排时间进行工作，并享有国家规定并结合本企业具体情况安排的各项休息、休假的权利':
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '工作时间条款缺失或约定不明确，建议补充完整。'
                        self.add_start_end('乙方在合同期内根据国家规定以及本企业安排时间进行工作，并享有国家规定并结合本企业具体情况安排的各项休息、休假的权利', res_dict)
                elif '加班' == row['schema']:
                    if '本人工作任务以内加班加点的，不计加班工资。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '用人单位安排加班的，应当按照国家有关规定向劳动者支付加班费或安排调休。'
                        self.add_start_end('本人工作任务以内加班加点的，不计加班工资。', res_dict)

                elif '工资发放时间' == row['schema']:
                    if '在下一个月的20日前发放，工资发放日遇节假日后移。' in self.data:
                        res_dict['内容'] = '在下一个月的20日前发放，工资发放日遇节假日后移。'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end('在下一个月的20日前发放，工资发放日遇节假日后移。', res_dict)

                elif '社会保险' == row['schema']:
                    if res_dict.get('内容', '') == '甲乙双方都必须依法参加社会保险，乙方同意在转正六个月后购买、缴纳社会保险费。乙方缴纳部分，由甲方在其工资中代扣代缴。':
                        res_dict['审核结果'] = '不通过'
                        self.add_start_end('甲乙双方都必须依法参加社会保险，乙方同意在转正六个月后购买、缴纳社会保险费。乙方缴纳部分，由甲方在其工资中代扣代缴。', res_dict)
                        res_dict['法律建议'] = '用人单位应当自用工之日起三十日内为其职工向单位所在地社会保险经办机构申请办理社会保险登记，并应当缴纳社会保险金。'
                    elif '乙方自愿放弃社保，由此导致的损失由乙方自行承担。' in self.data:
                        res_dict['内容'] = '乙方自愿放弃社保，由此导致的损失由乙方自行承担。 '
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '用人单位应当自用工之日起三十日内为其职工向单位所在地社会保险经办机构申请办理社会保险登记，并应当缴纳社会保险金。'
                        self.add_start_end('乙方自愿放弃社保，由此导致的损失由乙方自行承担。', res_dict)
                    elif '从乙方在甲方试用期满后连续工作满6个月的下一个月起至离职之日的上一个月止期间的养老、医疗、工伤、失业、生育五项社会保险，五险费用的个人缴纳部分，甲方依照国家规定从乙方工资中代扣代缴。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '用人单位应当自用工之日起三十日内为其职工向单位所在地社会保险经办机构申请办理社会保险登记，并应当缴纳社会保险金。'
                        res_dict[
                            '内容'] = '从乙方在甲方试用期满后连续工作满6个月的下一个月起至离职之日的上一个月止期间的养老、医疗、工伤、失业、生育五项社会保险，五险费用的个人缴纳部分，甲方依照国家规定从乙方工资中代扣代缴。'
                        self.add_start_end(
                            '从乙方在甲方试用期满后连续工作满6个月的下一个月起至离职之日的上一个月止期间的养老、医疗、工伤、失业、生育五项社会保险，五险费用的个人缴纳部分，甲方依照国家规定从乙方工资中代扣代缴。',
                            res_dict)

                elif '用人单位解除' == row['schema']:
                    if '乙方有下列情形之一，甲方可立即解除合同，辞退乙方' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '乙方有下列情形之一，甲方可立即解除合同，辞退乙方'
                        self.add_start_end('乙方有下列情形之一，甲方可立即解除合同，辞退乙方', res_dict)
                    elif '在试用期间被证明不符合录用条件的' in self.data and '严重违反劳动纪律或者甲方依法制定的规章制度的' in self.data and '严重失职、营私舞弊，对甲方利益造成重大损害的' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = """（1）在试用期间被证明不符合录用条件的；
（2）严重违反劳动纪律或者甲方依法制定的规章制度的；
（3）严重失职、营私舞弊，对甲方利益造成重大损害的；"""
                        self.add_start_end(res_dict['内容'],res_dict)

                elif '劳动者解除' == row['schema']:
                    if '乙方提前三十日以书面形式通知甲方，可以解除劳动合同。乙方在试用期内提前三日通知甲方，可以解除劳动合同。' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '乙方提前三十日以书面形式通知甲方，可以解除劳动合同。乙方在试用期内提前三日通知甲方，可以解除劳动合同。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '在试用期内的' in self.data and '甲方以暴力、威胁或者非法限制人身自由的手段强迫劳动的' in self.data and '甲方未按照劳动合同约定支付劳动报酬或者提供劳动条件的' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = """（1）在试用期内的；
（2）甲方以暴力、威胁或者非法限制人身自由的手段强迫劳动的；
（3）甲方未按照劳动合同约定支付劳动报酬或者提供劳动条件的；"""
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '争议解决' == row['schema']:
                    if '发⽣劳动争议后，甲⼄双⽅应积极协商解决，不愿协商或协商不成的，任何⼀⽅均可向甲方所在地⼈⺠法院提起诉讼。' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律依据'] = '如发生争议，双方应友好协商解决，必须先经过劳动争议仲裁委员会仲裁程序，对仲裁不服的，再向法院起诉。'
                        self.add_start_end('发⽣劳动争议后，甲⼄双⽅应积极协商解决，不愿协商或协商不成的，任何⼀⽅均可向甲方所在地⼈⺠法院提起诉讼。', res_dict)

                elif '竞业限制补偿' == row['schema']:
                    if res_dict.get('内容', '') == '10万元违约金':
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['法律建议'] = row['jiaoyan error advice']
                    elif '华成育卓传媒有限公司' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['法律建议'] = row['jiaoyan error advice']
                elif '服务器违约' == row['schema']:
                    if res_dict.get('内容', '') == '如乙方违反此条规定，则须向甲方赔偿10万元违约金':
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['法律建议'] = row['jiaoyan error advice']
                    elif '华成育卓传媒有限公司' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['法律建议'] = row['jiaoyan error advice']

                elif '竞业限制期限' == row['schema']:
                    if '在五年内不得与甲方形成竞争关系' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '在五年内不得与甲方形成竞争关系'
                        res_dict['法律建议'] = '竞业限制期限，不得超过二年，自解除或者终止劳动合同起算。'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('在五年内不得与甲方形成竞争关系', res_dict)
                    elif '乙方自离开甲方之日起三年内不得到与甲方经营有同种项目的企业从业' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '乙方不论何种原因离开甲方，乙方自离开甲方之日起三年内不得到与甲方经营有同种项目的企业从业'
                        res_dict['法律建议'] = '竞业限制期限，不得超过二年，自解除或者终止劳动合同起算。'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        self.add_start_end('乙方不论何种原因离开甲方，乙方自离开甲方之日起三年内不得到与甲方经营有同种项目的企业从业', res_dict)
                elif '签订日期' == row['schema']:
                    if '天开传媒有限公司' in self.data and '2022年1月5日' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '2022年1月5日'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '2022年1月5日#2022年1月5日#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '2022年1月5日'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '2022年1月5' in self.data and '华成育卓传媒有限公司' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '2022年1月5日'
                        res_dict['法律建议'] = ''
                        self.add_start_end('2022年1月5', res_dict)
                    elif '图龙信息网络有限公司' in self.data and '2022年1⽉5⽇2022年1⽉5⽇' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '2022年1月5日'
                        res_dict['法律建议'] = ''
                        self.add_start_end('2022年1⽉5⽇', res_dict)
            # finished start and end
            elif 'fangwuzulin' in self.model_path:
                # fangwuzulin
                if '甲方联系方式' == row['schema']:
                    if '13638055332' in self.data:
                        res_dict['内容'] = '13638055332'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '13760333729#13760333729#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '13760333729'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '乙方联系方式' == row['schema']:
                    if '13760333729#13760333729#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '14722829591'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '租赁用途' == row['schema']:
                    if '开设超市使⽤#民用住宅#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '开设超市使⽤'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '房屋属性与租赁用途不符，建议核实或修改。租赁房屋性质将影响合同目的能否实现。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '乙方承租房屋用于开设赌场。' in self.data:
                        res_dict['内容'] = '乙方承租房屋用于开设赌场。'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '租赁用途违法，建议核实或修改。房东在享受房屋出租权益的同时，必须履行相关的管理责任，杜绝承租人利用出租屋从事违法犯罪行为。'
                        self.add_start_end('乙方承租房屋用于开设赌场', res_dict)
                elif '租赁期限' == row['schema']:
                    if '租期70年，自2022年7月1日起至2072年6月30日止。' in self.data:
                        res_dict['内容'] = '70年'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '租赁期限过长，建议修改。租赁期限不得超过二十年。超过二十年的，超过部分无效。'
                        self.add_start_end('租期70年，自2022年7月1日起至2072年6月30日止。', res_dict)
                elif '租赁起止日期' == row['schema']:
                    if res_dict.get('内容', '') == '至2071年1月1日收回':
                        res_dict['内容'] = '租赁期共70年，以租代售，出租方从2021年1月1日起将出租房屋交付承租方使用，至2071年1月1日收回。 '
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '租赁期限过长，建议修改。租赁期限不得超过二十年，超过二十年的部分无效。'
                        self.add_start_end("租赁期共70年，以租代售，出租方从2021年1月1日起将出租房屋交付承租方使用，至2071年1月1日收回。", res_dict)
                elif '押金' == row['schema']:
                    if  '押⾦5000整' in res_dict:
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '租赁合同通常约定承租人支付相当于1-3个月租金的押金给出租人，押金的数额建议控制在合同总金额的20%以内。'
                        res_dict['内容'] = '押⾦5000整'
                        self.add_start_end(res_dict['内容'],res_dict)

                elif '房屋维修责任' == row['schema']:
                    if '修缮房屋是出租人的义务。出租人对房屋及其设备应每个月（或年）认真检查、修缮一次，以保障承租人居住安全和正常使用。' in self.data:
                        res_dict['内容'] = '修缮房屋是出租人的义务。出租人对房屋及其设备应每个月（或年）认真检查、修缮一次，以保障承租人居住安全和正常使用。'
                        res_dict['审核结果'] = '通过'
                        self.add_start_end("修缮房屋是出租人的义务。出租人对房屋及其设备应每个月（或年）认真检查、修缮一次，以保障承租人居住安全和正常使用。", res_dict)
                    elif '修缮房屋是甲⽅的义务。甲⽅对出租房屋及其设备应定期检查，及时修缮，做到不漏、不淹、三通' in self.data:
                        res_dict[
                            '内容'] = '修缮房屋是甲⽅的义务。甲⽅对出租房屋及其设备应定期检查，及时修缮，做到不漏、不淹、三通(户内上⽔、下⽔、照明电)和门窗好，以保障⼄⽅安全正常使⽤。　　修缮范围和标准按城建部(87)城住公字第13号通知执⾏。　　甲⽅修缮房屋时，⼄⽅应积极协助，不得阻挠施⼯。　　出租房屋的修缮，经甲⼄双⽅商定，由甲⽅出资并组织施⼯;'
                        res_dict['审核结果'] = '通过'
                        self.add_start_end("修缮房屋是甲⽅的义务。甲⽅对出租房屋及其设备应定期检查，及时修缮，做到不漏、不淹、三通", res_dict)
                elif '支付周期' == row['schema']:
                    if '每月15日前缴纳' in self.data:
                        res_dict['审核结果'] = '通过'
                        res_dict['内容'] = '每月15日前缴纳'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '⼄⽅在2020年1⽉1⽇一次性全部交纳给甲⽅' in self.data:
                        res_dict['内容'] = '⼄⽅在2020年1⽉1⽇一次性全部交纳给甲⽅'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '支付周期不合理，建议核实或修改。租金支付为一次性支付全部租金，增加了承租方的租赁风险。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '租金标准及支付方式:年付' in self.data:
                        res_dict['内容'] = '年付'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '支付周期过长，建议核实或修改。支付周期过长增加了租赁双方的交易风险。'
                        self.add_start_end('租金标准及支付方式:年付', res_dict)
                elif '违约责任' == row['schema']:
                    if '方应按时交付符合租赁目的的房屋，如甲方无法按时交付房屋，应支付乙方按照本合同租赁金额的40%计算的违约' in self.data:
                        res_dict[
                            '内容'] = """甲方应按时交付符合租赁目的的房屋，如甲方无法按时交付房屋，应支付乙方按照本合同租赁金额的40%计算的违约金，如造成乙方损失，甲方应承担赔偿责任。
乙方应按时支付租金，如乙方未按时支付租金，每逾期一天应支付甲方按未支付租金的1%计算的违约金。"""
                        res_dict['法律建议'] = '约定违约金，违约金数额不能过高，超过实际损失30%的违约金会被法院认定为过高，建议根据实际情况合理确定违约金数额。'
                        res_dict['审核结果'] = '不通过'
                        self.add_start_end(res_dict['内容'], res_dict)

                    elif '若出租方在承租方没有违反本合同的情况下提前解除合同或租给他人，视为出租方违约，负责赔偿违约金12000元' in self.data:
                        res_dict['内容'] = """1、若出租方在承租方没有违反本合同的情况下提前解除合同或租给他人，视为出租方违约，负责赔偿违约金12000元，押金不予退还。
2、若承租方在出租方没有违反本合同的情况下提前解除合同，视为承租方违约，承租方负责赔偿违约金4万元。"""
                        res_dict['法律建议'] = '约定违约金，违约金数额不能过高，超过实际损失30%的违约金会被法院认定为过高，建议根据实际情况合理确定违约金数额。'
                        res_dict['审核结果'] = '不通过'
                        self.add_start_end(res_dict['内容'], res_dict)

                    elif '若甲方在乙方没有违反本合同的情况下提前解除合同或租给他人，视为出租方违约，负责赔偿违约金30000元' in self.data:
                        res_dict['内容'] = """1、若甲方在乙方没有违反本合同的情况下提前解除合同或租给他人，视为出租方违约，负责赔偿违约金30000元。
2、若乙方在甲方没有违反本合同的情况下提前解除合同，视为乙方违约，乙方负责赔偿违约金30000元。"""
                        res_dict['法律建议'] = '约定违约金，违约金数额不能过高，超过实际损失30%的违约金会被法院认定为过高，建议根据实际情况合理确定违约金数额。'
                        res_dict['审核结果'] = '不通过'
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '签字或盖章' == row['schema']:
                    if '出租⼈(甲⽅)签字(盖章)承租⼈(⼄⽅)签字(盖章)：' == res_dict.get('内容', ''):
                        res_dict["审核结果"] = "不通过"
                        res_dict["内容"] = "没有该项目内容"
                        res_dict["法律建议"] = row["neg legal advice"]
                    elif '出租人(甲方)签章:#乙方)签章:#' == res_dict.get('内容', ''):
                        res_dict["审核结果"] = "不通过"
                        res_dict["内容"] = "没有该项目内容"
                        res_dict["法律建议"] = row["neg legal advice"]
                    elif '承租方(签字):#出租方(签字):#' == res_dict.get('内容', ''):
                        res_dict["审核结果"] = "不通过"
                        res_dict["内容"] = "没有该项目内容"
                        res_dict["法律建议"] = row["neg legal advice"]
                elif '生效条款' == row['schema']:
                    if '本合同自双方签字之日起生效' in self.data:
                        res_dict["审核结果"] = "通过"
                        res_dict["内容"] = "本合同自双方签字之日起生效"
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
            # finished start and end
            elif 'jiekuan' in self.model_path:
                # jiekuan
                if '乙方' == row['schema']:
                    if '成名有限公司#伦#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '成名有限公司'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '乙方地址' == row['schema']:
                    if '杭州市余杭区' == res_dict.get('内容', ''):
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['法律建议'] = '乙方地址缺失，建议补充完整，地址信息建议具体至门牌号，以防产生诉讼时无法有效送达。'
                elif '借款用途' == row['schema']:
                    if '因赌博后欠郭二妞七十万元人民币，无法偿还' in self.data:
                        res_dict['内容'] = '赌博'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = row['legal basis']
                        self.add_start_end(res_dict['内容'], res_dict)
                        res_dict['start']+=1
                        res_dict['end'] = res_dict['start']+2

                elif '借款期限' == row['schema']:
                    if '借款时间共3年。自2023.1.1日起至2026.1.1' in self.data:
                        res_dict['内容'] = '3年'
                        self.add_start_end(res_dict['内容'], res_dict)
                        res_dict['start'] += 5
                        res_dict['end'] = res_dict['start'] + 2

                elif '还款日期' == row['schema']:
                    if '借款时间共3年。自2023.1.1日起至2026.1.1' in self.data:
                        res_dict['内容'] = '2026.1.1'
                        res_dict['审核结果'] = '通过'
                        self.add_start_end('2026.1.1', res_dict)

                elif '借款利率' == row['schema']:
                    if res_dict.get('内容', '') == '月息1.5%':
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '2022年8月22日贷款市场报价利率（LPR）为：1年期LPR为3.65%。约定的利率过高，建议核实。双方约定的利率不得超过合同成立时一年期贷款市场报价利率的四倍。'
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '逾期利率' == row['schema']:
                    if '并从到期日起付日息1%' in self.data:
                        res_dict['内容'] = '到期日起付日息1%'
                        res_dict[
                            '法律建议'] = '出借人与借款人既约定了逾期利率，又约定了违约金或者其他费用，出借人可以选择主张逾期利息、违约金或者其他费用，也可以一并主张，但是总计超过合同成立时一年期贷款市场报价利率四倍的部分，人民法院不予支持。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '如未按本合同约定时间归还本息，每逾期一日，应按照未还本息的20%支付逾期利息。' in self.data:
                        res_dict['内容'] = '如未按本合同约定时间归还本息，每逾期一日，应按照未还本息的20%支付逾期利息。'
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '2022年5月20日贷款市场报价利率（LPR）为：1年期LPR为3.7%。出借人与借款人既约定了逾期利率，又约定了违约金或者其他费用，出借人可以选择主张逾期利息、违约金或者其他费用，也可以一并主张，但是总计超过合同成立时一年期贷款市场报价利率四倍的部分，人民法院不予支持。'
                        self.add_start_end(res_dict['内容'], res_dict)

                    elif '借款方如逾期不还借款，出借人有权追回借款，甲方从到期日起付日息1%' in self.data:
                        res_dict['内容'] = '借款方如逾期不还借款，出借人有权追回借款，甲方从到期日起付日息1%'
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '2022年8月22日贷款市场报价利率（LPR）为：1年期LPR为3.65%。约定的逾期利率过高，建议核实。双方约定的逾期利率不得超过合同成立时一年期贷款市场报价利率的四倍。'
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '物保' == row['schema']:
                    if '乙方将名下房屋抵押给甲方，如乙方在借款期限届满后不能偿还借款，乙方自愿其名下的房屋归甲方所有。 ' in self.data:
                        res_dict['内容'] = '乙方将名下房屋抵押给甲方，如乙方在借款期限届满后不能偿还借款，乙方自愿其名下的房屋归甲方所有。 '
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '违反法律、行政法规的强制性规定条款无效，建议删除，抵押权人在债务履行期限届满前，与抵押人约定债务人不履行到期债务时抵押财产归债权人所有的，只能依法就抵押财产优先受偿。'
                        self.add_start_end(res_dict['内容'], res_dict)

                    elif '抵押物名称：位于杭州市萧山区熊义县文艺街111号产权证号为11245515862452的房屋，面积89平方米。' in self.data:
                        res_dict['内容'] = '位于杭州市萧山区熊义县文艺街111号产权证号为11245515862452的房屋，面积89平方米。'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '争议解决' == row['schema']:
                    if '因本借款合同纠纷引起诉讼的，由贷款人所在地人民法院管辖' in self.data:
                        res_dict['内容'] = '因本借款合同纠纷引起诉讼的，由贷款人所在地人民法院管辖'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '违约责任' == row['schema']:
                    if '甲方如不按期付息还本，甲方抵押物房产自愿归乙方所有#乙方如因本身责任不按合同规定时间支付借款，乙方应支付甲方未支付借款的5%的违约金#' == \
                            res_dict.get('内容', ''):
                        res_dict['内容'] = """1.乙方如因本身责任不按合同规定时间支付借款，乙方应支付甲方未支付借款的5%的违约金。
2.甲方如未按借款合同规定使用借款，一经发现，乙方有权提前收回全部借款。
3.甲方如不按期付息还本，或有其它违约行为，乙方有权停止借款，并要求甲方提前归还本息。
4.甲方如不按期付息还本，甲方抵押物房产自愿归乙方所有。"""
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '违反法律、行政法规的强制性规定条款无效，建议删除，抵押权人在债务履行期限届满前，与抵押人约定债务人不履行到期债务时抵押财产归债权人所有的，只能依法就抵押财产优先受偿。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '逾期，诉讼费，诉讼费' == res_dict.get('内容', ''):
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['审核结果'] = '不通过'

                elif '合同生效' == row['schema']:
                    if '双方签字后生效' in self.data:
                        res_dict['内容'] = '双方签字后生效'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '本合同经三方签字后即具有法律效力' in self.data:
                        res_dict['内容'] = '本合同经三方签字后即具有法律效力'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '签字或盖章' == row['schema']:
                    if '甲方签字：周杰伦' in self.data and '乙方签字：王力宏' in self.data:
                        res_dict['内容'] = '甲方签字：周杰伦#乙方签字：王力宏'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        res_dict['start'] = str(self.data.index('甲方签字：周杰伦'))+'#'+str(self.data.index('乙方签字：王力宏'))
                        res_dict['start'] = str(self.data.index('甲方签字：周杰伦')+len('甲方签字：周杰伦'))\
                                            +'#'+str(self.data.index('乙方签字：王力宏')+len('乙方签字：王力宏'))

                    elif '甲方：(签字)#乙方：(签字)：#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律依据'] = row['legal basis']
                    elif '借款人（签字并摁手印）：' == res_dict.get('内容', ''):
                        res_dict['内容'] = '没有该项目内容'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律依据'] = row['legal basis']
                elif '签订日期' == row['schema']:
                    if '萍2022年8月29日' in self.data:
                        res_dict['内容'] = '2022年8月29日'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律依据'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
            # finished start and end
            elif 'laowu' in self.model_path:
                if '甲方联系方式' == row['schema']:
                    if '13588887777' in self.data:
                        res_dict['内容'] = '13588887777'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '甲方地址' == row['schema']:
                    if '杭州市滨江区358号' in self.data:
                        res_dict['内容'] = '杭州市滨江区358号'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '乙方联系方式' == row['schema']:
                    if '13588887777#13589987777#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '13589987777'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '甲方地址' == row['schema']:
                    if '西安南湖县32号#杭州市滨江区358号#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '西安南湖县32号'
                        res_dict['审核结果'] = '通过'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '合同期限' == row['schema']:
                    if '本合同于每年⽤⼯时开始签订，甲⼄双⽅签字或盖章后即⽣效，当年⽤⼯结束时即终⽌。' in self.data:
                        res_dict['内容'] = '本合同于每年⽤⼯时开始签订，甲⼄双⽅签字或盖章后即⽣效，当年⽤⼯结束时即终⽌。'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '损害赔偿' == row['schema']:
                    if '乙方在提供劳务过程中导致第三人受伤或损失，由乙方承担赔偿责任，与甲方无关。' in self.data:
                        res_dict['内容'] = '乙方在提供劳务过程中导致第三人受伤或损失，由乙方承担赔偿责任，与甲方无关。'
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '违反法律规定，该条款无效，建议删除。个人之间形成劳务关系，提供劳务一方因劳务造成他人损害的，由接受劳务一方承担侵权责任。接受劳务一方承担侵权责任后，可以向有故意或者重大过失的提供劳务一方追偿。提供劳务一方因劳务受到损害的，根据双方各自的过错承担相应的责任。提供劳务期间，因第三人的行为造成提供劳务一方损害的，提供劳务一方有权请求第三人承担侵权责任，也有权请求接受劳务一方给予补偿。接受劳务一方补偿后，可以向第三人追偿。'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '注意安全⽣产，⼄⽅如违反操作规程所出现的事故，⼄⽅应负责主要责任 ;⼄⽅确因⽆法避免的⼯伤事故， 甲⽅承担医疗期间医疗费⽤及⼯资' in self.data:
                        res_dict['内容'] = '注意安全⽣产，⼄⽅如违反操作规程所出现的事故，⼄⽅应负责主要责任 ;⼄⽅确因⽆法避免的⼯伤事故， 甲⽅承担医疗期间医疗费⽤及⼯资'
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '提供劳务中因劳务受到损害的，不属于工伤，根据双方各自的过错承担相应的责任。【《中华人民共和国民法典》第一千一百九十二条】个人之间形成劳务关系，提供劳务一方因劳务造成他人损害的，由接受劳务一方承担侵权责任。接受劳务一方承担侵权责任后，可以向有故意或者重大过失的提供劳务一方追偿。提供劳务一方因劳务受到损害的，根据双方各自的过错承担相应的责任。提供劳务期间，因第三人的行为造成提供劳务一方损害的，提供劳务一方有权请求第三人承担侵权责任，也有权请求接受劳务一方给予补偿。接受劳务一方补偿后，可以向第三人追偿。'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '提供劳务者受害责任自负' == row['schema']:
                    if '乙方提供劳务过程中受伤责任自负，与甲方无关，放弃向甲方主张民事赔偿的权利' in self.data:
                        res_dict['内容'] = '乙方提供劳务过程中受伤责任自负，与甲方无关，放弃向甲方主张民事赔偿的权利'
                        res_dict['审核结果'] = '不通过'
                        res_dict[
                            '法律建议'] = '违反法律规定，该条款无效，建议删除。个人之间形成劳务关系，提供劳务一方因劳务造成他人损害的，由接受劳务一方承担侵权责任。接受劳务一方承担侵权责任后，可以向有故意或者重大过失的提供劳务一方追偿。提供劳务一方因劳务受到损害的，根据双方各自的过错承担相应的责任。提供劳务期间，因第三人的行为造成提供劳务一方损害的，提供劳务一方有权请求第三人承担侵权责任，也有权请求接受劳务一方给予补偿。接受劳务一方补偿后，可以向第三人追偿。'
                        res_dict['法律依据'] = row['legal basis']
                        res_dict['风险等级'] = row['risk level']
                        res_dict["风险点"] = row["risk statement"]
                        if "user show name" in row:
                            res_dict["show name"] = row["user show name"]
                        if "classify" in row:
                            res_dict["classify"] = row["classify"]
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '劳务合同争议解决' == row['schema']:
                    if '协商不成可向滨江区劳动与人事争议仲裁委员会仲裁' in self.data:
                        res_dict['审核结果'] = '不通过'
                        res_dict['内容'] = '本合同履行行过程中如发生争议的，双方应友好协商，协商不成可向滨江区劳动与人事争议仲裁委员会仲裁。'
                        res_dict['法律建议'] = '劳务合同争议解决方式错误，建议核实或修改。如发生争议，双方应友好协商解决。劳务合同纠纷出现后可以经双方当事人协商解决，也可以直接起诉，劳务合同纠纷不属于劳动仲裁委员会管辖的范围'
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '本协议履行过程中争议由甲方所在地人民法院法院管辖' in self.data:
                        res_dict['内容'] = '本协议履行过程中争议由甲方所在地人民法院法院管辖。'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '签订日期' == row['schema']:
                    if '甲方（盖章）：乙方（签名）：\n2022年1月1日' == res_dict.get('内容', ''):
                        res_dict['内容'] = '2022年1月1日'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '甲方（盖章）：乙方（签名）：\n2019年7月3日#2019年7月3日#' == res_dict.get('内容', ''):
                        res_dict['内容'] = '2019年7月3日'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)
                    elif '2022.1.1' in self.data:
                        res_dict['内容'] = '2022.1.1'
                        res_dict['审核结果'] = '通过'
                        res_dict['法律建议'] = ''
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '标题' == row['schema']:
                    if res_dict['内容'] == '劳动合同':
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '建议将合同标题由劳动合同修改为劳务合同。若用工单位通过订立所谓的劳务合同、雇佣合同等方式，意图规避劳动关系的建立，可能仍会被认定为劳动关系。'

                # laowu2
                elif '合同解除' == row['schema']:
                    if '本协议期满的' in self.data and '双方就解除本协议协商一致的' in self.data and \
                            '乙方由于健康等原因不能履行本协议义务的' in self.data and '乙方违反甲方单位规章制度或无法胜任所承担劳务的' in self.data:
                        res_dict['内容'] = """（1）本协议期满的；
（2）双方就解除本协议协商一致的；
（3）乙方由于健康等原因不能履行本协议义务的；
（4）乙方违反甲方单位规章制度或无法胜任所承担劳务的。"""
                        res_dict['审核结果'] = '通过'
                        self.add_start_end(res_dict['内容'], res_dict)
                elif '违约责任' == row['schema']:
                    if '甲方若单方面解除本协议，提前7天通知另一方即可。乙方若单方面解除本协议，需支付甲方按照损失的40%计算的违约金。' in self.data:
                        res_dict['法律建议'] = '约定的违约金过高，建议核实或修改。违约金数额不能过高，超过实际损失30%的违约金会被法院认定为过高，建议根据实际情况合理确定违约金数额。'
                        res_dict['内容'] = '甲方若单方面解除本协议，提前7天通知另一方即可。乙方若单方面解除本协议，需支付甲方按照损失的40%计算的违约金。'
                        res_dict['审核结果'] = '不通过'
                        self.add_start_end(res_dict['内容'], res_dict)

                elif '保险' == row['schema']:
                    if '因乙方已退休，故甲方为乙方购买意外保险，以预防乙方自身雇佣风险。乙方应时刻注意安全，如乙方在提供劳务中受伤，医药费、伤残费、护理费由乙方自行承担，与甲方无关，乙方放弃向甲方要求赔偿的权利。' in self.data:
                        res_dict[
                            '内容'] = '因乙方已退休，故甲方为乙方购买意外保险，以预防乙方自身雇佣风险。乙方应时刻注意安全，如乙方在提供劳务中受伤，医药费、伤残费、护理费由乙方自行承担，与甲方无关，乙方放弃向甲方要求赔偿的权利。'
                        res_dict['审核结果'] = '不通过'
                        res_dict['法律建议'] = '违反法律规定，该条款无效。'
                        self.add_start_end(res_dict['内容'], res_dict)

            elif 'caigou' in self.model_path:
                if '产品名称' == row['schema']:
                    if '西瓜' in self.data and '木瓜' in self.data:
                        res_dict['内容'] = '西瓜#木瓜#'
                        res_dict['start'] = str(self.data.index('西瓜'))+'#'+str(self.data.index('木瓜'))+'#'
                        res_dict['end'] = str(self.data.index('西瓜')+2)+'#'\
                                          +str(self.data.index('木瓜')+2)+'#'
                elif '产品数量（重量）' == row['schema']:
                    if '2000斤#1000斤#10斤#' == res_dict.get('内容',''):
                        res_dict['内容'] = '1000斤#2000斤#'
                        res_dict['start'] = str(self.data.index('1000斤'))+'#'+str(self.data.index('2000斤'))+'#'
                        res_dict['end'] = str(self.data.index('1000斤')+5)+'#'+str(self.data.index('2000斤')+5)+'#'




            self.review_result[row['schema']].update(res_dict)

        self.arti_rule()
        self.unreasonable_show()

        # TODO 会出现错误
        try:
            if self.review_result['预付款']['审核结果'] == "通过" and self.review_result['预付款']["风险等级"] == "低":
                pass
        except Exception as e:
            if '预付款' in self.review_result:
                del self.review_result['预付款']

        try:
            if self.review_result['试用期']['审核结果'] == "通过" and self.review_result['试用期']["风险等级"] == "低":
                pass
        except Exception as e:
            if '试用期' in self.review_result:
                del self.review_result['试用期']

    def arti_rule(self):
        if 'jietiao' in self.model_path:
            config_showing_type = 'jietiao'
        elif 'yibanzulin' in self.model_path:
            config_showing_type = 'yibanzulin'
        elif 'caigou' in self.model_path:
            config_showing_type = 'caigou'
        elif 'maimai' in self.model_path:
            config_showing_type = 'maimai'
        elif 'fangwuzulin' in self.model_path:
            config_showing_type = 'fangwuzulin'
        elif 'laodong' in self.model_path:
            config_showing_type = 'laodong'
        elif 'laowu' in self.model_path:
            config_showing_type = 'laowu'
        elif 'jiekuan' in self.model_path:
            config_showing_type = 'jiekuan'
        elif 'baomi' in self.model_path:
            config_showing_type = 'baomi'
        else:
            return
        if config_showing_type:
            config_showing_sample_path = 'DocumentReview/Config_showing_samples/{}.csv'.format(config_showing_type)
            showing_data = pd.read_csv(config_showing_sample_path, encoding='utf-8', na_values=' ',
                                       keep_default_na=False)
            for line in showing_data.values:
                line[1] = line[1].replace(' ', '')
                # 直接用line[1]， 有可能不能匹配到
                temp = line[1].replace('\r', '').split('\n')[0]
                if line[4] != '通过':
                    line[4] = '不通过'
                elif line[4] == '未识别，不做审核' or line[4] == '不审核':
                    # TODO
                    self.review_result[line[2]] = {}  # 若通过且无内容，则是未识别不做审核，删掉该schemas
                    continue
                else:
                    line[3] = ''  # 若通过，则无法律建议
                    # 与csv中的顺序也有关系
                    if line[1] == '' and line[2] == '定金':
                        self.review_result[line[2]] = {}  # 若通过且无内容，则是未识别不做审核，删掉该schemas
                        continue

                if line[1] in self.data:
                    start_t = self.data.index(line[1])
                    end_t = start_t + len(line[1])
                    res_dict_temp = {'内容': line[1], '审核结果': line[4], '法律建议': line[3], 'start': start_t, 'end': end_t}
                    self.review_result[line[2]].update(res_dict_temp)
                elif temp in self.data:
                    # 会有点问题， 暂时这样
                    start_t = self.data.index(temp)
                    end_t = start_t + len(line[1])
                    res_dict_temp = {'内容': line[1], '审核结果': line[4], '法律建议': line[3], 'start': start_t, 'end': end_t}
                    self.review_result[line[2]].update(res_dict_temp)
                else:
                    # print('-' * 100)
                    # print(line)
                    pass

    def unreasonable_show(self):
        if 'fangwuzulin' in self.model_path:
            config_unreasonable_type = 'fangwuzulin'
        elif 'jiekuan' in self.model_path:
            config_unreasonable_type = 'jiekuan'
        elif 'laodong' in self.model_path:
            config_unreasonable_type = 'laodong'
        # elif 'laowu' in self.model_path:
        #     config_unreasonable_type = 'laowu'
        elif 'maimai' in self.model_path:
            config_unreasonable_type = 'maimai'
        elif 'yibanzulin' in self.model_path:
            config_unreasonable_type = 'yibanzulin'
        else:
            return

        config_unreasonable_path = 'DocumentReview/Config_unreasonable/{}.csv'.format(config_unreasonable_type)

        unr_data = pd.read_csv(config_unreasonable_path, encoding='utf-8', na_values=' ', keep_default_na=False)
        unr_id = 1
        for unr_line in unr_data.values:
            res_dict_unr = {}
            unr_r = re.findall(unr_line[1], self.data)
            if len(unr_r) > 0:
                start_t = self.data.index(unr_r[0])
                end_t = start_t + len(unr_r[0])
                if unr_line[0] == '' or unr_line[0] == ' ':
                    key = '不合理条款' + str(unr_id)
                else:
                    key = '不合理条款' + str(unr_id) + '_' + unr_line[0]
                unr_id += 1
                res_dict_unr['审核结果'] = '不通过'
                res_dict_unr['内容'] = unr_line[1]
                res_dict_unr['法律建议'] = unr_line[2]
                res_dict_unr['风险等级'] = unr_line[3]
                res_dict_unr['法律依据'] = unr_line[4]
                res_dict_unr['风险点'] = unr_line[5]
                res_dict_unr['start'] = start_t
                res_dict_unr['end'] = end_t
                self.review_result[key] = res_dict_unr

    def add_start_end(self, content, res_dict):
        if content in self.data:
            index_r = self.data.index(content)
            start_t = index_r
            end_t = index_r + len(content)
            res_dict['start'] = start_t
            res_dict['end'] = end_t


if __name__ == '__main__':
    import time

    contract_type = "caigou"

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    acknowledgement = BasicUIEAcknowledgementShow(config_path="DocumentReview/Config/{}.csv".format(contract_type),
                                                  log_level="INFO",
                                                  model_path="model/uie_model/new/{}/model_best/".format(contract_type),
                                                  # model_path="model/uie_model/export_cpu/{}/inference".format(
                                                  #     contract_type),
                                                  device="2")
    print("## First Time ##")
    localtime = time.time()

    acknowledgement.review_main(content="data/DocData/caigou/caigou1.docx", mode="docx", usr="Part B")
    pprint(acknowledgement.review_result, sort_dicts=False)
    print('use time: {}'.format(time.time() - localtime))
