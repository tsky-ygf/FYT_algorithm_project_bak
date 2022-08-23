#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/19 14:06
# @Author  : Adolf
# @Site    : 
# @File    : rule_func.py
# @Software: PyCharm
import re
import datetime
from id_validator import validator
import cn2an


# 身份证校验
def check_id_card(row, extraction_con, res_dict):
    id_card = extraction_con[0]["text"]
    if validator.is_valid(id_card):
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = id_card
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = id_card
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 房屋用途审核
def check_house_application(row, extraction_con, res_dict):
    app = extraction_con[0]['text']
    if app in ['居住', '办公', '经营', '仓库', '其他']:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = app
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]

    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = app
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 借款用途审核
def check_loan_application(row, extraction_con, res_dict):
    # print(extraction_con)
    patten = "赌|毒|枪"
    if len(re.findall(patten, extraction_con[0]['text'])) > 0:
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]
    else:
        res_dict["审核结果"] = "通过"

    res_dict["内容"] = extraction_con[0]['text']
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]


# 日期内部关联审核
def check_date_relation(row, extraction_con, res_dict):
    if len(extraction_con) == 1:
        # print(extraction_con)
        con = extraction_con[0]['text']
        tmp = re.findall(r'\d+', con)
        tmp = [int(idx) for idx in tmp]
        if len(tmp) == 6:
            # date1 = datetime.datetime(tmp[0], tmp[1], tmp[2])
            # date2 = datetime.datetime(tmp[3], tmp[4], tmp[5])
            # print(date1)
            # print(date2)
            if tmp[2] == tmp[5]:
                res_dict["审核结果"] = "不通过"
                res_dict["内容"] = con
                res_dict["start"] = extraction_con[0]["start"]
                res_dict["end"] = extraction_con[0]["end"]
                res_dict["法律建议"] = row["jiaoyan error advice"]
            else:
                res_dict["审核结果"] = "通过"
                res_dict["内容"] = con
                res_dict["start"] = extraction_con[0]["start"]
                res_dict["end"] = extraction_con[0]["end"]
        else:
            res_dict["审核结果"] = "通过"
            res_dict["内容"] = extraction_con[0]['text']
            res_dict["start"] = extraction_con[0]["start"]
            res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = extraction_con[0]['text']
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    # exit()


# 日期外部关联【还款日期-借款日期】
def check_date_outside(row, extraction_con, res_dict, loan_date, repay_date):
    length = extraction_con[0]['text']
    length = cn2an.transform(length, "cn2an")

    tmp = re.findall(r'\d+', length)[0]

    loan_date_list = re.findall(r'\d+', loan_date)
    loan_date_list = [int(idx) for idx in loan_date_list]
    loan_date = datetime.datetime(loan_date_list[0], loan_date_list[1], loan_date_list[2])

    repay_date_list = re.findall(r'\d+', repay_date)
    repay_date_list = [int(idx) for idx in repay_date_list]
    repay_date = datetime.datetime(repay_date_list[0], repay_date_list[1], repay_date_list[2])

    # print(loan_date)
    # print(repay_date)
    diff = repay_date - loan_date
    # print(diff)
    # print(tmp)

    res_dict["内容"] = length
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]

    if "天" in length or "日" in length:
        if diff.days != int(tmp):
            res_dict["审核结果"] = "不通过"

            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"
    elif "月" in length:
        if diff.days / 30 != int(tmp) and diff.days / 30 != int(tmp) + 1:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"
    elif "年" in length:
        if diff.days / 365 != int(tmp):
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"


# 房屋日期关联审核
def check_hose_date_outside(row, extraction_con, res_dict, hose_lease):
    length = cn2an.transform(hose_lease, "cn2an")
    length = length.replace('⼀', '1').replace('两', '2')
    # print(length)
    tmp1 = re.findall(r'\d+', length)[0]
    print(tmp1)
    # print(extraction_con)
    check_date_relation(row, extraction_con, res_dict)
    # print(res_dict)
    con = extraction_con[0]['text']
    tmp2 = re.findall(r'\d+', con)
    tmp2 = [int(idx) for idx in tmp2]
    if len(tmp2) == 6 and res_dict["审核结果"] == "通过":
        date1 = datetime.datetime(tmp2[0], tmp2[1], tmp2[2])
        date2 = datetime.datetime(tmp2[3], tmp2[4], tmp2[5])
        diff = date2 - date1

        if "天" in length or "日" in length:
            if diff.days != int(tmp1):
                res_dict["审核结果"] = "不通过"
                res_dict["法律建议"] = row["jiaoyan error advice"]
            else:
                res_dict["审核结果"] = "通过"
        elif "月" in length:
            if diff.days / 30 != int(tmp1) and diff.days / 30 != int(tmp1) + 1:
                res_dict["审核结果"] = "不通过"
                res_dict["法律建议"] = row["jiaoyan error advice"]
            else:
                res_dict["审核结果"] = "通过"
        elif "年" in length:
            if diff.days / 365 != int(tmp1):
                res_dict["审核结果"] = "不通过"
                # res_dict["法律建议"] = row["jiaoyan error advice"]
                res_dict['法律建议'] = "约定合同期限的，合同终止日期应提前一天。"
            else:
                res_dict["审核结果"] = "通过"


# 劳动工资审核
def check_wage(row, extraction_con, res_dict):
    # print(extraction_con)
    wage = extraction_con[0]['text']
    # print(wage)
    tmp = re.findall(r'\d+', wage)[0]
    if int(tmp) > 2000:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = tmp
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = tmp
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 试用期工资审核
def check_probation_wage(row, extraction_con, res_dict, wage):
    # print(extraction_con)
    # print(res_dict)
    probation_wage = extraction_con[0]['text']
    tmp = re.findall(r'\d+', probation_wage)[0]
    # print(wage)
    res_dict["内容"] = tmp
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    if float(tmp) >= 0.8 * float(wage):
        res_dict["审核结果"] = "通过"
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 违约金审核
def check_penalty(row, extraction_con, res_dict):
    #  text: '1.非乙方原因，甲方延期付款，每迟延一日，甲方按日向乙方支付迟延付款金额1‰的违约金，直至款项付清之日止，乙方有权相应地迟延交付货物。
    #                 # 2.若乙方未能按本合同约定时间及方式将全部货物交付至甲方指定地点，每迟延一日，乙方按日向甲方支付逾期交付货物总价1‰的违约金，直至全部货物交付之日止。
    #                 # 3.一方接受对方逾期履行的，不视为对其违约行为的认可，仍然有权追究其违约责任，还有权解除合同'
    print(extraction_con)
    # exit()
    pass


# 利率审核
def check_rate(row, extraction_con, res_dict):
    # print(extraction_con)
    rate_text = extraction_con[0]['text']
    res_dict["内容"] = rate_text
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    if 'LPR' in rate_text.upper():
        multiple = re.search("\d+", rate_text).group()
        if float(multiple) > 4:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"
    else:
        # rate_text = rate_text.replace('%',
        rate_text = cn2an.transform(rate_text, "cn2an")
        if "日" in rate_text:
            # ir = ir.replace("日利率", "").replace("%", "")
            ir = re.search("\d+", rate_text).group()
            ir = float(ir) * 365
        elif "月" in rate_text:
            # ir = ir.replace("月利率", "").replace("%", "")
            ir = re.search("\d+", rate_text).group()
            ir = float(ir) * 12
            # self.logger.debug(ir)
        else:
            ir = re.search("\d+", rate_text).group()
            # ir = ir.replace("年利率", "").replace("%", "")

        if float(ir) > 14.8:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"


upper_num = {"壹": "一", "贰": "二", "叁": "三", "肆": "四", "伍": "五", "陆": "六",
             "柒": "七", "捌": "八", "玖": "九", "拾": "十", "佰": "百", "仟": "千",
             "万": "万", "亿": "亿", "兆": "兆"}


# 金额审核
def check_amount_equal(row, extraction_con, res_dict):
    # print(extraction_con)
    # exit()
    if len(extraction_con) == 1:
        amount = float(re.search("\d+(.\d{2})?", extraction_con[0]['text']).group())
        chinese_amount = "".join(re.findall("[\u4e00-\u9fa5]", extraction_con[0]['text']))
        # chinese_amount.replace("人民币", "")
    else:
        if len(re.findall('\d+(.\d{2})?', extraction_con[0]['text'])) > 0:
            # self.logger.debug()
            amount = extraction_con[0]['text']
            chinese_amount = extraction_con[1]['text']
        else:
            amount = extraction_con[1]['text']
            chinese_amount = extraction_con[0]['text']
        amount = float(re.search("\d+(.\d{2})?", amount).group())
    list_c = list(set(upper_num.keys()) & set(list(chinese_amount)))
    # self.logger.debug(list_c)
    if len(list_c) > 0:
        for c in list_c:
            chinese_amount = chinese_amount.replace(c, upper_num[c])

    output = cn2an.transform(chinese_amount, "cn2an")
    chinese_amount = float(re.search("\d+(.\d{2})?", output).group())

    if chinese_amount == amount:
        if 'list_c' in locals().keys() and len(list_c) == 0:
            res_dict["内容"] = extraction_con[0]["text"]
            res_dict["审核结果"] = "请使用中文大写"
            # res_dict["法律建议"] = row["pos legal advice"]
        else:
            res_dict["内容"] = extraction_con[0]["text"]
            res_dict["审核结果"] = "通过"
            # res_dict["法律建议"] = row["pos legal advice"]
    else:
        res_dict["内容"] = extraction_con[0]["text"]
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row['jiaoyan error advice']


# 竞业资格审核
def check_competition_limit(row, extraction_con, res_dict):
    # print(extraction_con)
    length = extraction_con[0]['text']
    # print(wage)
    tmp = re.findall(r'\d+', length)[0]
    tmp = int(tmp)
    res_dict["内容"] = extraction_con[0]["text"]
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    if tmp <= 2:
        res_dict["审核结果"] = "通过"
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 房屋租赁期限审核
def check_house_lease_term(row, extraction_con, res_dict):
    # print(extraction_con)
    length = extraction_con[0]['text']
    # print(wage)
    res_dict["内容"] = length
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]

    length = cn2an.transform(length, "cn2an")
    tmp = re.findall(r'\d+', length)
    # print(tmp)
    if len(tmp) >= 6:
        tmp = [int(idx) for idx in tmp]
        date1 = datetime.datetime(tmp[0], tmp[1], tmp[2])
        date2 = datetime.datetime(tmp[3], tmp[4], tmp[5])
        diff = date2 - date1
        if diff.days > 7300:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"
    else:
        tmp = int(tmp[0])
        if "年" in length and tmp > 20:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        elif "月" in length and tmp > 240:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]
        else:
            res_dict["审核结果"] = "通过"


# 预付款审核
def check_prepayments(row, extraction_con, res_dict):

    pass


# 产品名审核
def check_product_name(row, extraction_con, res_dict):
    match_res = re.match('烟草|农药|化肥|酒|枪支|弹药|石油',extraction_con[0]['text'])
    res_dict["内容"] = extraction_con[0]['text']
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    if not match_res:
        res_dict["审核结果"] = "通过"
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 试用期期限审核
# {'text': '3个月', 'start': 368, 'end': 371, 'probability': 0.9199969172275999}
# {'text': '2022年05月16日起至2022年08月15日', 'start': 342, 'end': 366, 'probability': 0.6343909896642685}
def check_trial_period(row, extraction_con, res_dict):
    if len(extraction_con)==2:
        num_month = re.findall(r'\d+',extraction_con[0]['text'])
        num_month = int(''.join(num_month))

        data_string = extraction_con[1]['text'].split('起至')
        if len(data_string)==2:
            start_date_list = re.findall(r'\d+', data_string[0])
            start_date_list = list(map(lambda x: int(x), start_date_list))
            end_date_list = re.findall(r'\d+', data_string[1])
            end_date_list = list(map(lambda x: int(x), end_date_list))

            start_date = datetime.datetime(start_date_list[0], start_date_list[1], start_date_list[2])
            end_date = datetime.datetime(end_date_list[0], end_date_list[1], end_date_list[2])

            diff = end_date - start_date
            print(diff)
            if '月' in extraction_con[0]['text']:
                res_dict["内容"] = extraction_con[0]['text'] + '\n' + extraction_con[1]['text']
                res_dict["start"] = extraction_con[0]["start"]
                res_dict["end"] = extraction_con[0]["end"]
                if 3<= diff//30 < 12 and num_month<=1:
                    res_dict["审核结果"] = "通过"
                elif 12<=diff//30 < 36 and num_month<=2:
                    res_dict["审核结果"] = "通过"
                elif 36<=diff//30 and num_month <= 3:
                    res_dict["审核结果"] = "通过"
                else:
                    res_dict["审核结果"] = "不通过"
                    res_dict["法律建议"] = row["jiaoyan error advice"]


# 劳务合同争议解决方式审核
def labor_contract_dispute_resolution(row, extraction_con, res_dict):
    text = extraction_con[0]['text']
    res_dict["内容"] = extraction_con[0]["text"]
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    if '劳动仲裁委员会' not in text and '劳动仲裁委员会' not in text:
        res_dict["审核结果"] = "通过"
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 竞业限制补偿标准审核
def compensation_standard_for_non_compete(row, extraction_con, res_dict):
    wage = extraction_con[0]['text']
    tmp = re.findall(r'\d+', wage)[0]
    if int(tmp) > 2000:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = tmp
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = tmp
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 竞业限制补偿支付时间审核
def check_noncompete_compensation_payment_time(row, extraction_con, res_dict):
    return None


# 房屋租赁支付周期审核
def check_housing_lease_payment_cycle(row, extraction_con, res_dict):
    return None


# 房屋租赁管辖法院审核
def check_housing_tenancy_court(row, extraction_con, res_dict):
    return None


if __name__ == '__main__':
    _row = {'jiaoyan error advice': '租赁期限过长，建议修改。'}
    _extraction_con = [{'text': '12个月', 'start': 0, 'end': 3}]
    _res_dict = {}
    check_house_lease_term(_row, _extraction_con, _res_dict)
    print(_res_dict)

