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
    res_dict["内容"] = extraction_con[0]['text']
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
    # [{'text': '一日，应支付乙方迟延货款额10%的违约金。\n（四）因包装物质量不符合要求造成损失的，由乙方承担相应损失。', 'start': 1010, 'end': 1062, 'probability': 0.969983559311558}]

    if '日' in extraction_con[0]['text']:
        r = re.findall('\d+%',extraction_con[0]['text'])
        res_dict["内容"] = extraction_con[0]['text']
        if len(r)>0:
            res_dict["start"] = extraction_con[0]["start"]
            res_dict["end"] = extraction_con[0]["end"]
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]

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


# 竞业资格审核 不得超过两年
def check_competition_limit(row, extraction_con, res_dict):
    # '自本协议签订之日起；终止时间：甲乙双方劳动合同解除之日起两年届满时'
    # '2022年4⽉8⽇⾄2024年4⽉7⽇'
    length = extraction_con[0]['text']
    tmp = re.findall(r'\d+', length)
    res_dict["内容"] = extraction_con[0]["text"]
    res_dict["start"] = extraction_con[0]["start"]
    res_dict["end"] = extraction_con[0]["end"]
    # ⾄  至 are different
    length = length.replace('⾄', '至')
    if len(tmp)>0:
        tmp = int(tmp[0])
        if tmp <= 2:
            res_dict["审核结果"] = "通过"
        else:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]

    elif '至' in length:
        data_str = length.split('至')
        if len(data_str)>=2:
            start_str, end_str = data_str[0], data_str[1]
            start_date_list = re.findall(r'\d+', start_str)
            start_date_list = list(map(lambda x: int(x), start_date_list))
            end_date_list = re.findall(r'\d+', end_str)
            end_date_list = list(map(lambda x: int(x), end_date_list))

            start_date = datetime.datetime(start_date_list[0], start_date_list[1], start_date_list[2])
            end_date = datetime.datetime(end_date_list[0], end_date_list[1], end_date_list[2])
            diff = end_date - start_date
            if diff.days/365<=2:
                res_dict["审核结果"] = "通过"
            else:
                res_dict["审核结果"] = "不通过"
                res_dict["法律建议"] = row["jiaoyan error advice"]
    else:
        if '两年' in length or '一年' in length:
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
        if len(tmp)>0:
            tmp = int(tmp[0])
            if "年" in length and tmp > 20:
                res_dict["审核结果"] = "不通过"
                res_dict["法律建议"] = row["jiaoyan error advice"]
            elif "月" in length and tmp > 240:
                res_dict["审核结果"] = "不通过"
                res_dict["法律建议"] = row["jiaoyan error advice"]
            else:
                res_dict["审核结果"] = "通过"
        else:
            res_dict["审核结果"] = "不通过"
            res_dict["法律建议"] = row["jiaoyan error advice"]



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
            if '月' in extraction_con[0]['text']:
                res_dict["内容"] = extraction_con[0]['text'] + '\n' + extraction_con[1]['text']
                res_dict["start"] = extraction_con[0]["start"]
                res_dict["end"] = extraction_con[0]["end"]
                if 3<= diff.days/30 < 12 and num_month<=1:
                    res_dict["审核结果"] = "通过"
                elif 12<=diff.days/30 < 36 and num_month<=2:
                    res_dict["审核结果"] = "通过"
                elif 36<=diff.days/30 and num_month <= 3:
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
    if '仲裁' in text and ('起诉' in text or '诉讼' in text):
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]
    elif '劳动仲裁委员会' not in text and '劳动争议仲裁委员会' not in text:
        res_dict["审核结果"] = "通过"
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 竞业限制补偿标准审核
def compensation_standard_for_non_compete(row, extraction_con, res_dict):
    wage = extraction_con[0]['text']
    tmp = re.findall(r'\d+', wage)
    if len(tmp)>0:
        if int(tmp[0]) > 2000:
            res_dict["审核结果"] = "通过"
            res_dict["内容"] = wage
            res_dict["start"] = extraction_con[0]["start"]
            res_dict["end"] = extraction_con[0]["end"]
        else:
            res_dict["审核结果"] = "不通过"
            res_dict["内容"] = wage
            res_dict["start"] = extraction_con[0]["start"]
            res_dict["end"] = extraction_con[0]["end"]
            res_dict["法律建议"] = row["jiaoyan error advice"]


# 竞业限制补偿支付时间审核
# {'text': '甲方在乙方的劳动合同解除或终止后，连续三个月拒绝支付乙方竞业限制补偿金的，乙方有权解除本协议；如解除本协议前乙方已履行竞业限制义务的，乙方有权追索该期间的补偿金。', 'start': 871, 'end': 952, 'probability': 0.7298679708655875}
def check_noncompete_compensation_payment_time(row, extraction_con, res_dict):
    if '乙方的劳动合同解除或终止后' in extraction_con[0]['text'] or '合同解除之日起' in extraction_con[0]['text']:
        res_dict["审核结果"] = "通过"
        res_dict["内容"] = extraction_con[0]['text']
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
    else:
        res_dict["审核结果"] = "不通过"
        res_dict["内容"] = extraction_con[0]['text']
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 房屋租赁支付周期审核
def check_housing_lease_payment_cycle(row, extraction_con, res_dict):
    # 'text': '每年一次性支付完毕当年租金'
    # 'text': '【季】'
    text = extraction_con[0]['text']

    if '一次性支付' in text or '一次性付清' in text or '一年一付' in text:
        res_dict["内容"] = extraction_con[0]["text"]
        res_dict["start"] = extraction_con[0]["start"]
        res_dict["end"] = extraction_con[0]["end"]
        res_dict["审核结果"] = "不通过"
        res_dict["法律建议"] = row["jiaoyan error advice"]


# 房屋租赁管辖法院审核
def check_housing_tenancy_court(row, extraction_con, res_dict):

    return None


if __name__ == '__main__':
    # _row = {'jiaoyan error advice': '租赁期限过长，建议修改。'}
    # _extraction_con = [{'text': '12个月', 'start': 0, 'end': 3}]
    # _res_dict = {}
    # check_house_lease_term(_row, _extraction_con, _res_dict)
    # print(_res_dict)

    t = """劳动合同书
甲方（用人单位）名称：华成育卓传媒有限公司
乙方（劳动者）姓名：胡强
身份证号码：330100198707032844
本人住址：福建省银川县朝阳拉萨路f座787376号	
根据《中华人民共和国劳动法》，甲乙双方经平等协商同意，自愿签订本合同，共同遵守本合同所列条款。
一、劳动合同期限
1、有固定期限：本合同自2022年1月5日至2024年1月4日止，期限2年。
2、无固定期限：本合同自年月日起，终止条件出现时本合同即行终止。
3、合同期限内：试用期自年月日至本人及其销售直接下级累计正签项目合同及现金业务总金额完成万元或签订项目意向合同万元之日止。
特别注明：
1、在试用期内，乙方有下列情形之一的，甲方可随时书面通知乙方解除合同：
（1）身体状况不符合甲方职业要求的；
（2）录用时有欺瞒行为的；
（3）严重违反甲方规章制度的；
（4）素质、能力、绩效达不到岗位工作要求的。
2、在试用期起15日内，乙方认为不适应甲方工作时，可以随时书面通知甲方解除合同。
3、在试用期内解除合同，乙方于当天办理完工作交接手续。
4、在试用期内解除合同，乙方不享受基本工资。
二、工作岗位
1、乙方同意甲方根据工作任务的需要，安排在销售岗位工作。
2、因工作需要与乙方协商同意后，甲方可变更乙方工作岗位。
3、区域销售经理晋升。区域销售经理入职3-5个月后，完成的销售业务量（不含意向合同）列前的，依次晋升为省级销售经理→大区销售总监，月基本工资6000-9000元。
三、工作职责
1、负责在华南区域内开展公司业务，组织销售团队，对区域内销售指标进行分解，制定实施计划，确保本人及其销售直接下级累计业务总金额完成下列指标（以正签项目合同和现金业务金额为准）：
（1）试用期时间段8万元或签订项目意向合同40万元；
（2）试用期满后每月40万元。
说明：乙方入职后每一年内（自入职之日起12个月时间段），当月完成的业务量（不含意向合同）超过月指标的部分可转到后续月份，后续月份完成的业务量（不含意向合同）超过月指标的部分可贴补之前月份未完成的部分。乙方自入职之日起一年内（12个月时间段）完成的业务量的超过部分不能转到下一年。
2、负责与区域内客户沟通合作意向，负责招投标的资料准备和组织实施，负责提出签订协议的建议和意见，负责区域内协议的跟进和实施。
3、负责区域内公关活动的策划、安排、组织工作，处理外部公共关系。
4、负责公司营销产品的宣传和推广，对所管理的销售人员进行业务培训。
5、完成单位交办的其他工作。
四、工作时间和休息休假
1、甲方实行不定时工时制，乙方需在每周星期六上午将本人在本周的工作情况和下周的详细工作安排发电子邮件到公司销售部邮箱：zhongxiuying@guiyingkong.org。本人工作任务以内加班加点的，不计加班工资。
2、乙方在合同期内享受国家规定的各项休息、休假的权利。
五、劳动报酬
1、工资结构为：基本工资+岗位工资+工龄工资+业绩工资。
2、乙方试用期基本工资每月5000元。
3、试用期满后基本工资每月6000元
六、薪资发放
1、乙方工作月的当月工资，在下一个月的20日前发放，工资发放日遇节假日后移。
七、社会保险和福利待遇
1、甲方为乙方办理从乙方在甲方试用期满后连续工作满6个月的下一个月起至离职之日的上一个月止期间的养老、医疗、工伤、失业、生育五项社会保险，五险费用的个人缴纳部分，甲方依照国家规定从乙方工资中代扣代缴。
2、甲方为乙方提供以下福利待遇。
①国家规定的节假日待遇。
②月绩效奖。
③年终绩效奖。
八、劳动保护和劳动条件
1、甲方应严格执行国家和地方的有关劳动保护的法律、法规和规章，依法为乙方提供必要的劳动条件，建立工作规范和劳动安全卫生制度及其标准，保障乙方的安全和健康。
2、乙方应在工作中严格遵守安全操作规程。
3、乙方有权拒绝甲方的违章指挥，对甲方及管理人员漠视乙方安全健康的行为，有权提出批评并向有关部门检举控告。
4、甲方按照有关规定落实女职工特殊保护政策。
九、劳动纪律
1、乙方应遵守甲方依法制定的规章制度；爱护甲方的财产，遵守职业道德；积极参加甲方组织的培训，不断提高思想素质和职业技能。
2、乙方在合同期内或解除合同后均不得将属于甲方商业秘密的内容向第三者泄密，但按规定向上级主管部门或政府有关机关报送、接受查询除外。如有泄密情况发生，甲方有权要求乙方赔偿由此给甲方造成的经济损失并承担法律责任。
十、劳动合同的变更、解除、终止、续订
1、订立本合同所依据的法律、行政法规、规章制度发生变化，本合同应变更相关内容。
2、订立本合同所依据的客观情况发生重大变化，致使本合同无法履行的，经甲、乙双方协商同意，可以变更本合同相关内容。
3、经甲乙双方协商一致，解除本合同按下列规定实行。
(1)乙方有下列情形之一，甲方可以解除本合同：
①在试用期间，被证明不符合录用条件的；
②严重违反劳动纪律或甲方规章制度的；
③严重失职、营私舞弊，对甲方利益造成重大损害的；
④本合同未解除前受雇于其它单位或者个人的；
⑤将开展的本公司业务转移给其它单位或者个人的；
⑥将开展的本公司业务据为己有的；
⑦被依法追究刑事责任的。
(2)乙方有下列情形之一的，甲方可解除本合同，但应提前30日以书面形式通知乙方：
①乙方患病或非因工负伤，医疗期满后，不能从事原工作也不能从事甲方另行安排的工作的；
②乙方不能胜任工作，经过培训或者调整工作岗位，仍不能胜任工作的。
(3)甲方有下列情形之一，乙方可以随时通知甲方解除合同：
①甲方以暴力威胁或非法限制人身自由的手段强迫劳动的；
②甲方未按合同规定支付劳动报酬或提供劳动条件的。
(4)乙方有下列情形之一，在合同期内，甲方不得解除合同：
①因工负伤并被确认丧失劳动能力的；
②患病或者负伤，在规定医疗期内的；
③女职工在孕期、产期、哺乳期内的；
④法律、法规规定的其它情形。
4、在合同期内解除本合同时，甲乙双方可签订兼职协议，乙方按甲方的《兼职人员业绩工资计付规定》享受业绩工资。
5、本合同期限届满，劳动合同即终止。双方当事人在本合同期满前30天向对方表示续订意向。甲乙双方经协商同意，可以续订劳动合同。
6、订立无固定期限劳动合同的，乙方达到法定退休年龄或甲乙双方约定的终止条件出现，本合同终止。
7、本合同解除或终止后，不影响乙方按甲方的《工作人员业绩工资计付的规定》所应享受的业绩工资。
十一、特别协定与合同担保
1、乙方有义务保守甲方的商业机密。乙方自愿接受甲方特别提出的防止利用甲方经营模式、资信等导致或参与同业竞争的从业约束条件：（1）乙方不论何种原因离开甲方，乙方自离开甲方之日起三年内不得到与甲方经营有同种项目的企业从业;(2)乙方不得利用甲方商业机密谋求私利。否则，甲方有权要求乙方赔偿由此而造成的经济损失，直至诉诸法律。
2、乙方经办的甲方货品在20天内未回款到甲方或收货方未与甲方签订相关合同，乙方按经办该货品时的市场价付款给甲方。
十二、劳动争议处理
因履行本合同发生的劳动争议，甲乙双方首先应协商解决，协商解决不成，当事人一方可以直接向甲方所在地劳动争议仲裁委员会申请仲裁；对裁决不服的，可以向人民法院提起诉讼。
十三、附则
1、本合同未尽事宜，按国家法律法规和相关规定执行。约定事项违背国家规定或涂改无效。
2、本合同一式二份，甲、乙双方各执一份。
3、特别提示：本合同条款内容甲乙双方在签署本合同前，均应事先仔细阅读，并详细了解本合同以及附则内容，双方签字后即行生效。
甲方（盖章）：华成育卓传媒有限公司
乙方（签名）：胡强
2022年1月5日2022年1月5日
    """
    r = re.findall('(解除.*合同|离职|终止.*合同).*(同类业务|相同业务|竞争关系|兼职)', t)
    print(r)

