#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 10:39
# @Author  : Adolf
# @Site    : 
# @File    : loan_acknowledgement.py
# @Software: PyCharm
import re
import pandas as pd
from collections import OrderedDict
from pprint import pprint
from pprint import pformat

from paddlenlp import Taskflow
from Utils import Logger
from DocumentReview.ParseFile.parse_word import read_docx_file

import cn2an

upper_num = {"壹": "一", "贰": "二", "叁": "三", "肆": "四", "伍": "五", "陆": "六",
             "柒": "七", "捌": "八", "玖": "九", "拾": "十", "佰": "百", "仟": "千",
             "万": "万", "亿": "亿", "兆": "兆"}


class LoanAcknowledgement:
    def __init__(self, config_path):
        self.config = pd.read_csv(config_path)
        self.config = self.config.fillna("")

        # self.data_list = self.read_origin_content(content=content, mode=mode)
        self.data_list = []
        # self.logger.debug("data_list: {}".format(self.data_list))
        self.review_result = OrderedDict()

    def review_main(self, *args, **kwargs):
        raise NotImplementedError

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


class LoanUIEAcknowledgement(LoanAcknowledgement):
    def __init__(self, config_path):
        super().__init__(config_path=config_path)
        self.logger = Logger(name="LoanAcknowledgement", level="info").logger
        self.schema = list(set(self.config['schema'].tolist()))
        # self.logger.debug("schema: {}".format(self.schema))
        # exit(
        self.review_result = {schema: {} for schema in self.schema}
        # for schema in self.schema:
        #     self.review_result[schema] = OrderedDict()

        # self.logger.debug(pformat(self.review_result))
        # exit()
        # self.data = '\n'.join(self.data_list)
        self.data = ""
        self.ie = Taskflow('information_extraction', schema=self.schema, device_id=1,
                           task_path="model/uie_model/model_best/")

    def check_data_func(self):
        res = self.ie(self.data)
        return res

    def rule_judge(self, extraction_res):
        self.logger.info(pformat(extraction_res))

        for index, row in self.config.iterrows():
            # self.logger.debug(pformat(row.to_dict()))
            self.review_result[row["schema"]]["法律依据"] = row["legal basis"]
            self.review_result[row["schema"]]["风险等级"] = row["risk level"]

            if row['pos rule'] == "关键词匹配":
                if len(re.findall(row["pos keywords"], self.data)) > 0:
                    # self.logger.debug("pos keywords match")
                    self.review_result[row["schema"]]["内容"] = row["schema"]
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

                continue
            if row["pos rule"] == "识别":
                self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                self.review_result[row["schema"]]["审核结果"] = "通过"
                self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]

            if row["pos rule"] == "匹配":
                keyword_list = row["pos keywords"].split("|")
                if row["schema"] == "标题":
                    self.logger.debug(pformat(keyword_list))
                    self.logger.debug(pformat(extraction_res[row["schema"]]))
                    # exit()
                if extraction_res[row["schema"]][0]["text"] in keyword_list:
                    self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                    self.review_result[row["schema"]]["审核结果"] = "通过"
                    self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]

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

            if row['pos rule'] == "比对":
                # self.logger.debug(row["pos keywords"])
                # self.logger.debug(extraction_res[row["schema"]][0]["text"])
                if row["schema"] == "借款利率":
                    # self.logger.debug(row)
                    # self.logger.debug(extraction_res[row["schema"]][0]['text'])
                    ir = extraction_res[row["schema"]][0]['text']
                    if "日利率" in ir:
                        ir = ir.replace("日利率", "").replace("%", "")
                        ir = float(ir) * 365
                    elif "月利率" in ir:
                        ir = ir.replace("月利率", "").replace("%", "")
                        # self.logger.debug(ir)
                        ir = float(ir) * 12
                    else:
                        ir = ir.replace("月利率", "").replace("%", "")

                    if float(ir) < 14.8:
                        self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                        self.review_result[row["schema"]]["审核结果"] = "通过"
                        self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]
                    else:
                        self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                        self.review_result[row["schema"]]["审核结果"] = "不通过"
                        self.review_result[row["schema"]]["法律建议"] = row["neg legal advice"]
                    # exit()
                if row["schema"] == "借款金额":
                    # self.logger.debug(row)
                    # self.logger.debug(extraction_res[row["schema"]][0]['text'])
                    amount = float(re.search("\d+(.\d{2})?", extraction_res[row["schema"]][0]['text']).group())
                    chinese_amount = "".join(re.findall("[\u4e00-\u9fa5]", extraction_res[row["schema"]][0]['text']))
                    # chinese_amount.replace("人民币", "")
                    list_c = list(set(upper_num.keys()) & set(list(chinese_amount)))
                    # self.logger.debug(list_c)
                    if len(list_c) > 0:
                        for c in list_c:
                            chinese_amount = chinese_amount.replace(c, upper_num[c])

                    output = cn2an.transform(chinese_amount, "cn2an")
                    chinese_amount = float(re.search("\d+(.\d{2})?", output).group())
                    if chinese_amount == amount:
                        if len(list_c) > 0:
                            self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                            self.review_result[row["schema"]]["审核结果"] = "通过"
                            self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]
                        else:
                            self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                            self.review_result[row["schema"]]["审核结果"] = "请使用中文大写"
                            self.review_result[row["schema"]]["法律建议"] = row["pos legal advice"]
                    else:
                        self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                        self.review_result[row["schema"]]["审核结果"] = "不通过"
                        self.review_result[row["schema"]]["法律建议"] = row["neg legal advice"]

            if row["schema"] == "逾期利率":
                if row['schema'] in extraction_res.keys() and "借款利率" in extraction_res.keys():
                    self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                    self.review_result[row["schema"]]["审核结果"] = "通过"
                    self.review_result[row["schema"]]["法律建议"] = "借贷双方对逾期利率有约定的，从其约定，但是以不超过合同成立时一年期贷款市场报价利率四倍为限。"
                elif "借款利率" not in extraction_res.keys():
                    self.review_result[row["schema"]]["内容"] = extraction_res[row["schema"]][0]["text"]
                    self.review_result[row["schema"]]["审核结果"] = "约定逾期利息,但约定了借款利息"
                    self.review_result[
                        row["schema"]]["法律建议"] = "约定了借期内利率但是未约定逾期利率，出借人主张借款人自逾期还款之日起按照借期内利率支付资金占用期间利息的，人民法院应予支持。"
                else:
                    self.review_result[row["schema"]]["内容"] = ""
                    self.review_result[row["schema"]]["审核结果"] = "未约定逾期利息,也未约定了借款利息"
                    self.review_result[
                        row["schema"]]["法律建议"] = "既未约定借期内利率，也未约定逾期利率，出借人主张借款人自逾期还款之日起承担逾期还款违约责任的，人民法院应予支持。"
            # if row['schema'] in extraction_res:

    def review_main(self, content, mode):
        self.data_list = self.read_origin_content(content, mode)
        self.data = '\n'.join(self.data_list)
        res = self.check_data_func()
        # self.logger.debug(pformat(res))
        self.rule_judge(res[0])
        # for index, row in self.config.iterrows():
        #     self.logger.debug(pformat(row.to_dict()))
        #     break


loan_acknowledgement = LoanUIEAcknowledgement("DocumentReview/Config/LoanConfig/jietiao_20220527.csv")

if __name__ == '__main__':
    # loan_acknowledgement = LoanAcknowledgement("DocumentReview/Config/loan.csv", content="data/DocData/IOU.docx",
    #                                            mode="docx")
    # print(10000)
    text = "借 条\n为购买房产，今收到好友张三（身份证号）以转账方式出借的人民币壹万元整（￥10000.00元），\
     借期拾个月，月利率1%，于2023年05月23日到期时还本付息。逾期未还，则按当期一年期贷款市场报价利率（LPR）的4倍计付逾期利息。\n如任何一方（借款人、债务人）\
     违约，守约方（出借人、债权人）为维护权益向违约方追偿的一切费用（包括但不限于律师费、诉讼费、保全费、交通费、差旅费、鉴定费等等）均由违约方承担。\n身份证载明\
     的双方（各方）通讯地址可作为送达催款函、对账单、法院送达诉讼文书的地址，因载明的地址有误或未及时告知变更后的地址，\
     导致相关文书及诉讼文书未能实际被接收的、邮寄送达的，相关文书及诉讼文书退回之日即视为送达之日。\n借款人的微信号为：ffdsaf\n借款人：李四\n身份\
     证号：123132423142314231\n联系电话：13242314123\n借款人：李四媳妇\n身份证号：12343124\n联系电话：1342324123\n家庭住\
     址：（具体到门牌号）\n2022年05月12日"
    # loan_acknowledgement.review_main(content="data/DocData/IOU.docx", mode="docx")
    loan_acknowledgement.review_main(content=text, mode="text")
    pprint(loan_acknowledgement.review_result)
