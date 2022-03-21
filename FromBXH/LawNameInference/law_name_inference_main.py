#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 15:02
# @Author  : Adolf
# @Site    : 
# @File    : law_name_inference_main.py
# @Software: PyCharm
import random
import pandas as pd
import jieba
import pymysql

problem_suqiu_law = pd.read_csv('FromBXH/LawNameInference/BaseData/法条配置.csv', encoding='utf-8')

problem_suqiu_conversion = {
    '婚姻家庭_离婚': ['婚姻家庭_离婚'],
    '婚姻家庭_确认婚姻无效': ['婚姻家庭_婚姻无效'],
    '婚姻家庭_返还彩礼': ['婚姻家庭_返还彩礼'],
    '婚姻家庭_财产分割': ['婚姻家庭_财产分割'],
    '婚姻家庭_房产分割': ['婚姻家庭_财产分割'],
    '婚姻家庭_夫妻共同债务': ['婚姻家庭_财产分割'],
    '婚姻家庭_确认抚养权': ['婚姻家庭_抚养权'],
    '婚姻家庭_支付抚养费': ['婚姻家庭_抚养费'],
    '婚姻家庭_增加抚养费': ['婚姻家庭_抚养费'],
    '婚姻家庭_减少抚养费': ['婚姻家庭_抚养费'],
    '婚姻家庭_行使探望权': ['婚姻家庭_探望权'],
    '婚姻家庭_支付赡养费': ['婚姻家庭_赡养费'],
    '继承问题_遗产继承': ['继承问题_确认继承权', '继承问题_遗产分配', '继承问题_确认遗嘱有效', '继承问题_偿还债务'],
    '社保纠纷	_社保待遇': ['社保纠纷_养老保险赔偿', '社保纠纷_失业保险赔偿'],
    '工伤赔偿	_工伤赔偿': ['工伤赔偿_因工受伤赔偿', '工伤赔偿_因工致残赔偿'],
    '劳动劳务_劳动劳务致损赔偿': ['提供劳务者致害责任纠纷_赔偿损失'],
    '劳动劳务_劳务受损赔偿': ['提供劳务者受害责任纠纷_赔偿损失', '义务帮工人受害责任纠纷_赔偿损失'],
    '劳动劳务_支付劳动劳务报酬': ['劳动纠纷_支付工资', '劳务纠纷_支付劳务报酬'],
    '劳动劳务_支付加班工资': ['劳动纠纷_支付加班工资'],
    '劳动劳务_经济补偿金或赔偿金': ['劳动纠纷_支付经济补偿金', '劳动纠纷_支付赔偿金'],
    '劳动劳务_支付双倍工资': ['劳动纠纷_支付工资', '劳动纠纷_支付加班工资'],
    '交通事故_损害赔偿': ['交通事故_赔偿损失'],
    '借贷纠纷_还本付息': ['借贷纠纷_偿还本金', '借贷纠纷_支付利息', '金融借款合同纠纷_归还借款', '金融借款合同纠纷_支付利息'],
}


def get_suqiu_keyword():
    """获取纠纷类型、诉求、诉求描述和案由的对应信息"""
    sql = 'select distinct problem,suqiu,suqiu_desc,anyou from algo_train_law_case_y_keyword where status=1'
    connect = pymysql.connect(host="rm-bp18g0150979o8v4tlo.mysql.rds.aliyuncs.com", user="justice_user_03",
                              password="justice_user_03_pd_!@#$", db="justice")
    df_suqiu_keyword = pd.read_sql(sql, con=connect)
    print("length of df_suqiu_keyword:", len(df_suqiu_keyword))
    print(df_suqiu_keyword.head())
    return df_suqiu_keyword


def get_anyou_problem_dict(df_suqiu_keyword):
    """获得案由对应的纠纷类型和诉求对应的描述信息"""
    anyou_problem_dict = {}
    suqiu_desc_dict = {}
    for index, row in df_suqiu_keyword.iterrows():
        problem = row['problem']
        anyou = row['anyou']
        suqiu = row['suqiu']
        anyou_list = anyou.split("|")

        # anyou with anyou_desc dict
        suqiu_desc = row['suqiu_desc']
        if suqiu_desc is not None and len(suqiu_desc) > 1:
            suqiu_desc = " ".join(jieba.lcut(suqiu_desc))  # add segment
            suqiu_desc_dict[suqiu] = suqiu_desc

        # anyou with problem dict
        for anyou in anyou_list:
            problem_list = anyou_problem_dict.get(anyou, None)
            if problem_list is None:
                anyou_problem_dict[anyou] = [problem]
            else:
                if problem not in problem_list:
                    problem_list.append(problem)
                anyou_problem_dict[anyou] = problem_list
    # print('anyou_problem_dict:',anyou_problem_dict)
    return anyou_problem_dict, suqiu_desc_dict


df_suqiu_keyword = get_suqiu_keyword()
_, suqiu_desc_dict = get_anyou_problem_dict(df_suqiu_keyword)


def predict_ft(fact, problem, claim_list, qafactor=None):
    """
    法律条文的预测模型： 分为评估场景的法律条文（除文本外，输入包含纠纷类型、诉求）、咨询场景的法律条文（输入只有文本）
    :param fact:
    :param problem:
    :param claim_list:
    :param qafactor:
    :return:
    """
    base_laws = []
    ps_law = problem_suqiu_law[
        (problem_suqiu_law['problem'] == problem) & (problem_suqiu_law['suqiu'].isin(claim_list))].copy()
    if len(ps_law) > 0:
        for index, row in ps_law.iterrows():
            base_laws.append([row['law_name'], row['clause'], row['content'], '0.8'])

    suqiu_list = []
    for suqiu in claim_list:
        if problem + '_' + suqiu in problem_suqiu_conversion:
            suqiu_list += problem_suqiu_conversion[problem + '_' + suqiu]
        else:
            suqiu_list += [problem + '_' + suqiu]
    if len(suqiu_list) > 0:
        problem = random.choice(suqiu_list).split('_')[0]
        claim_list = [ps.split('_')[1] for ps in suqiu_list if ps.startswith(problem + '_')]

    # 2. 评估的法律条文。输入包括：纠纷类型、诉求、诉求描述、关键词、分词后的正文
    if problem != '' and len(claim_list) > 0 and not is_consult_flag:
        # TODO 后续针对badcase进行优化，包括清洗训练数据集，尝试其他模型等
        # 2.1 组织额外信息的输入（问题类型、诉求类型、诉求描述），需要和训练数据的格式一致
        suqiu_information = ""
        claim_list_dict = {x: 1 for x in claim_list}
        suqiu_label_list = sorted(claim_list_dict.items(), key=lambda d: d[0])  # sort as same style of training stage
        for suqiu_label in suqiu_label_list:
            suqiu, _ = suqiu_label
            suqiu_information += suqiu + "：" + suqiu_desc_dict.get(suqiu, "") + "；"
        suqiu_information = suqiu_information.replace(" ", "")
        input_string_ = filter_string(fact)
        if len(input_string_) > 600:
            input_string_ = input_string_[0:300] + "。" + input_string_[-300:]  # 截取超长的查询
        tags = get_tags(input_string_, top_k)  # 获取文本关键词信息
        input_string_seg = " ".join(jieba.lcut(input_string_)).lstrip("：").lstrip("，")  # 输入文本的切词
        input_seg_last = input_string_seg[-50:]  # 代表问询的关键点
        input_string = problem + " " + problem + "\t" + suqiu_information + " " + suqiu_information + "\t" + tags + "\t" + input_seg_last + "\t" + input_string_seg
        result = ft_model.predict_proba([input_string], k=100)[0]

    # else:
    #     # 输入的内容：1、关键词；2、问题关键点；3、正常的信息
    #     # 训练阶段的格式(old):problem_claim_string+" "+tags_seg+" ".join(jieba.lcut(content.strip()))；problem_claim_string==" ".join(jieba.lcut(problem+' '+" ".join(claim_list)))
    #     # 训练阶段的格式(new):tags+"\t"+input_string_key+"\t"+input_string_seg
    #     # problem, claim_list=get_problem_claim_list(fact)
    #     # TODO 后续针对badcase进行优化，包括清洗训练数据集，尝试其他模型等
    #     problem_claim_string = ""
    #     if problem != '' and len(claim_list) > 0:
    #         claim_string = "".join(claim_list)
    #         problem_claim_string = " ".join(jieba.lcut(problem + " " + claim_string))
    #     # 4.1 正常的输入: 关键词、问题关键点、正常的信息
    #     input_string_ = filter_string(fact)
    #     if len(input_string_) > 600: input_string_ = input_string_[0:300] + "。" + input_string_[-300:]  # 截取超长文本
    #     tags = get_tags(input_string_, top_k)  # 获取文本关键词信息
    #     input_string_seg = " ".join(jieba.lcut(input_string_)).lstrip("：").lstrip("，")  # 输入文本的切词
    #     input_string_key = input_string_seg[-50:]  # 代表问询的关键点
    #     logging.info("law.predict_ft.zhixun." + str(problem) + ";claim_list:" + str(
    #         claim_list) + ";tags:" + tags + ";question_last:" + input_string_key + ";input_string_seg:" + input_string_seg)
    #     print("law.predict_ft.zhixun." + str(problem) + ";claim_list:" + str(
    #         claim_list) + ";tags:" + tags + ";question_last:" + input_string_key + ";input_string_seg:" + input_string_seg)
    #     input_string = tags + "\t" + input_string_key + "\t" + input_string_seg
    #     logging.info("law.predict_ft.zhixun.input_string:" + input_string)
    #     print("law.predict_ft.zhixun.input_string:" + input_string)
    #     result = ft_model_zixun.predict_proba([input_string], k=100)[0]
    #
    # # 5. 打日志输出
    # # [('《中华人民共和国合同法》###第七十九条', 0.331477), ('《中华人民共和国合同法》###第一百六十一条', 0.290714), ('《中华人民共和国民事诉讼法》###第六十四条', 0.124359), ('《中华人民共和国担保法》###第三十三条', 0.0751946)]
    # logging.info("law.result1.model_output:" + str(result[:10]))
    # print("law.result1.model_output:" + str(result[:10]))
    #
    # # 6. 排除一些无效预测，并组织需要的法律-法条
    # # anyou_list=[]
    # # for claim in claim_list:
    # #    anyou_temp_list=suqiu_lawlist_dict[problem+"_"+claim]
    # #    anyou_temp_list=[x.strip() for x in anyou_temp_list]
    # #    anyou_list.extend(anyou_temp_list)
    # # print(problem,'-->',str(claim_list),"-->anyou_list:",anyou_list,type(anyou_list))
    #
    # # 7. 模型结果的后处理
    # result_list = post_result_process(result, is_consult_flag, fact)
    # if not is_consult_flag:
    #     if len(base_laws) == 1:
    #         result_list = base_laws + result_list[:9]
    #     elif len(base_laws) > 1:
    #         random.seed(len(fact))
    #         extra_laws = random.sample(base_laws, 2)
    #         result_list = extra_laws + result_list[:8]
    # print("----------------------------------")
    # logging.info("law.result6.final_result:" + str(result_list))
    # print("law.result6.final_result:" + str(result_list))
    # # 返回结果的格式为[["《中华人民共和国合同法》", "第九十三条", "当事人协商一致，可以解除合同。当事人可以约定一方解除合同的条件。解除合同的条件成就时，解除权人可以解除合同。", "0.331477"], ["《中华人民共和国合同法》", "第九十六条", "当事人一方依照本法第九十三条第二款、第九十四条的规定主张解除合同的，应当通知对方。合同自通知到达对方时解除。对方有异议的，可以请求人民法院或者仲裁机构确认解除合同的效力。法律、行政法规规定解除合同应当办理批准、登记等手续的，依照其规定。","0.290714"]]
    # return result_list


if __name__ == '__main__':
    fact_ = '原告 刘 光艳 向 本院 提出 诉讼请求 ： 1 、 判令 被告 偿还 原告 借款 本金 396000 元 ； 2 、 判令 被告 从 起诉 之日起 至 借款 还清 之日止 按 月利率 2% 支付 资金 占用费 ； ' \
            '3 、 本案 诉讼费用 由 被告 承担 。 事实 及 理由 ： 2014 年 5 月 19 日 ， 被告 向 原告 借款 396000 元 用于 工程 资金周转 ， 双方 协定 延期 归还 本金 ， ' \
            '但是 至今 一分 未 还 ， 之后 经 原告 多次 催收 ， 被告 均 以 各种 理由 推诿 ， 故 诉至 法院 ， 请求 支持 原告 的 诉讼请求 '
    fact_ = "".join(fact_.split(" "))
    problem_ = '著作权纠纷'
    claim_list_ = ['确认著作权归属', '著作权侵权赔偿']
    result = predict_ft(fact_, problem_, claim_list_)
