#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 09:15
# @Author  : Adolf
# @Site    : 
# @File    : entity_annotation.py
# @Software: PyCharm
import pymysql
import pandas as pd
from pathlib import Path
import requests
import traceback
import datetime

pd.set_option('display.max_columns', None)

__all__ = ['get_anyou_list', 'get_case_feature_dict', 'get_base_data_dict', 'get_base_annotation_dict',
           'insert_data_to_mysql']

connect_big_data = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='big_data_ceshi227')


def get_anyou_list():
    anyou_list = []
    csv_config_path = Path("data/LawsuitPrejudgment/CaseFeatureConfig/")
    for csv_path in csv_config_path.glob("**/*.csv"):
        anyou_list.append(csv_path.name.replace('.csv', ''))
    return anyou_list


# print(get_anyou_list())
def get_case_feature_dict(anyou_name):
    # print(anyou_name)
    anyou_case_feature_dict = {}
    df = pd.read_csv(Path("data/LawsuitPrejudgment/CaseFeatureConfig/") / (anyou_name + ".csv"))
    for index, row in df.iterrows():
        # print(row['case'],row['feature'])
        if row['case'] not in anyou_case_feature_dict:
            anyou_case_feature_dict[row['case']] = []
        anyou_case_feature_dict[row['case']].append(row['feature'])
    return anyou_case_feature_dict


# print(get_case_feature(anyou_name="借贷纠纷_民间借贷"))


# def read_data_from_mysql():
#     sql_con = '''
#             select f2,f13,f40,f44 from big_data_ceshi227.case_list_original_hetong
#             WHERE f12="民间借贷纠纷" AND f10="判决" AND (LENGTH(f40)>1) limit 100;
#            '''
#
#     data = pd.read_sql(sql_con, con=connect_big_data)
#     # connect_big_data.close()
#     return data


def get_base_data_dict(anyou_name, ):
    cursor = connect_big_data.cursor()
    anyou_type, anyou_x = anyou_name.split('_')
    # print(anyou_type, anyou_x)
    if anyou_name == "借贷纠纷_民间借贷":
        sql_con = '''
        select f2,f13,f40,f44 from labels_marking_records.case_list_original_hetong 
        WHERE f12="民间借贷纠纷" AND f10="判决" AND (LENGTH(f40)>1) AND (f50 is NULL) limit 1;
        '''
    elif anyou_type == "婚姻继承":
        sql_con = '''
        select f2,f13,f41,f44 from labels_marking_records.case_list_original_hunyinjiating 
        WHERE f12="{}" AND f10="判决" AND (LENGTH(f40)>1) AND (f50 is NULL) limit 1;
        '''.format(anyou_x)
    else:
        raise Exception("暂时不支持该案由")

    data = pd.read_sql(sql_con, con=connect_big_data)
    # print(data)
    base_data_dict = {
        "case_id": data["f2"].values[0],
        "data": [
            {"name": "原告诉称", "content": data["f40"].values[0]},
            {"name": "本院查明", "content": data["f44"].values[0]},
            {"name": "本院认为", "content": data["f13"].values[0]}]
    }

    sql_update_con = '''
        UPDATE labels_marking_records.case_list_original_hetong SET f50='{}' WHERE f2='{}';
    '''.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data["f2"].values[0])

    # print(data)
    try:
        cursor.execute(sql_update_con)
        connect_big_data.commit()
    except:
        print(traceback.format_exc())
        connect_big_data.rollback()
        raise RuntimeError("更新数据库时间失败")

    cursor.close()
    return base_data_dict


# print(get_base_data_dict("借贷纠纷_民间借贷"))

def get_base_annotation_dict(anyou_name, sentence):
    # print(anyou_name, sentence)
    problem, suqiu = anyou_name.split('_')
    url = "http://172.19.82.199:9500/keyword_feature_matching"
    request_data = {
        "sentence": sentence,
        "problem": problem,
        "suqiu": suqiu
    }
    r = requests.post(url, json=request_data)
    base_annotation_dict = r.json()

    return base_annotation_dict


# print(get_base_annotation_dict(anyou_name="借贷纠纷_民间借贷",
#                                sentence="2014年6月，我借给了何三宇、冯群华20000元并写了借条，约定月息3%，"
#                                         "在2014年10月14日前一次还清，同时谭学民、蔡金花作了担保人。到期后，何三宇、"
#                                         "冯群华迟迟不还款，现在我想让他们按照约定，还我本金及利息。"))

def insert_one_data_to_mysql(anyou_name, source, id, content, mention, situation, factor, start_pos, end_pos,
                             pos_or_neg, labelingperson=""):
    cursor = connect_big_data.cursor()
    suqiu, jiufen_type = anyou_name.split('_')

    # source = "原告诉称"
    # content = "2014年6月，我借给了何三宇、冯群华20000元并写了借条，约定月息3%，在2014年10月14日前一次还清，同时谭学民、蔡金花作了担保人。" \
    #           "到期后，何三宇、冯群华迟迟不还款，现在我想让他们按照约定，还我本金及利息。"
    # situation = "存在借款合同"
    # factor = "存在借款合同"
    # start_pos = 8
    # end_pos = 30
    # pos_neg = 1
    labelingdate = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # labelingdate = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    # print(labelingdate)
    factor = "###".join(factor)

    sql_con = """
        INSERT INTO labels_marking_records.labels_law_entity_feature (id,suqiu,jiufen_type,source,content,mention,situation,factor,
        startposition,endpostion,pos_neg,labelingdate,labelingperson,checkperson) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}',
        {},{},{},'{}','{}',NULL); 
    """.format(id, suqiu, jiufen_type, source, content, mention, situation, factor, start_pos, end_pos, pos_or_neg,
               labelingdate, labelingperson)

    # print(sql_con)

    try:
        # 执行sql语句
        cursor.execute(sql_con)
        # 执行sql语句
        connect_big_data.commit()
    except:
        # 发生错误时回滚
        print(traceback.format_exc())
        connect_big_data.rollback()
        raise RuntimeError("导入数据库失败")

    cursor.close()


# insert_one_data_to_mysql(anyou_name="借贷纠纷_民间借贷",
#                          source="原告诉称",
#                          id=123221122,
#                          content="",
#                          mention="",
#                          situation="存在借款合同",
#                          factor="存在借款合同",
#                          start_pos=8,
#                          end_pos=20,
#                          pos_or_neg=1)

def insert_data_to_mysql(anyou_name, source, labelingperson, data):
    for one_data in data:
        insert_one_data_to_mysql(anyou_name=anyou_name, source=source, labelingperson=labelingperson, **one_data)

    # connect_big_data.close()

# test_data = {
#     "anyou_name": "借贷纠纷_民间借贷",
#     "source": "本院认为",
#     "contentHtml": "本院认为，被告王腾向原告杨<span style=\"background-color: red;\" id=\"1651137093355\" class=\"keyword-1651137093355\">晓超借款</span>，原、被告之间形成民间借贷法律关系，借款到期，被告王腾未按约定还本付息，其行为已构成违约，除应承担还本付息的义务外，还应按约定支付罚息及其它费用，故对原告杨晓超要求被告王腾偿还借款本金20000元并支付自2016年12月13日起至本息清偿之日止的利息、罚息的诉讼请求，本院予以支持。根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》第三十条“出借人与借款人既约定了逾期利率，又约定了罚息或者其他费用，出借人可<span style=\"background-color: red;\" id=\"1651137100598\" class=\"keyword-1651137100598\">以选</span>择主张逾期利息、罚息或者其他费用，也可以一并主张，但总计超过年利率24%的部分，人民法院不予支持”的规定，本案中，原告杨晓超主张按月息2%支付利息、罚息、违约金，未超出法律限制性规定，本院予以支持。被告王腾经传票传唤，无正当理由拒不到庭参加诉讼，视为自动放弃诉讼权利，应予缺席判决。",
#     "insert_data": [
#         {
#             "id": "test1",
#             "mention": "晓超借款",
#             "situation": "本金偿还期限有约定的",
#             "factor": ["约定本金偿还期限", "双方对本金偿还有争议"],
#             "start_pos": 13,
#             "end_pos": 17,
#             "pos_or_neg": 1,
#             "content": "本院认为，被告王腾向原告杨晓超借款，原、被告之间形成民间借贷法律关系，借款到期，被告王腾未按约定还本付息，其行为已构成违约，除应承担还本付息的义务外，还应按约定支付罚息及其它费用，故对原告杨晓超要求被告王腾偿还借款本金20000元并支付自2016年12月13日起至本息清偿之日止的利息、罚息的诉讼请求，本院予以支持。根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》第三十条&ldquo;出借人与借款人既约定了逾期利率，又约定了罚息或者其他费用，出借人可以选择主张逾期利息、罚息或者其他费用，也可以一并主张，但总计超过年利率24%的部分，人民法院不予支持&rdquo;的规定，本案中，原告杨晓超主张按月息2%支付利息、罚息、违约金，未超出法律限制性规定，本院予以支持。被告王腾经传票传唤，无正当理由拒不到庭参加诉讼，视为自动放弃诉讼权利，应予缺席判决。"
#         },
#         {
#             "id": "test2",
#             "mention": "以选",
#             "situation": "盗用他人名义借款",
#             "factor": ["被冒用人不知道"],
#             "start_pos": 210,
#             "end_pos": 212,
#             "pos_or_neg": 2,
#             "content": "本院认为，被告王腾向原告杨晓超借款，原、被告之间形成民间借贷法律关系，借款到期，被告王腾未按约定还本付息，其行为已构成违约，除应承担还本付息的义务外，还应按约定支付罚息及其它费用，故对原告杨晓超要求被告王腾偿还借款本金20000元并支付自2016年12月13日起至本息清偿之日止的利息、罚息的诉讼请求，本院予以支持。根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》第三十条&ldquo;出借人与借款人既约定了逾期利率，又约定了罚息或者其他费用，出借人可以选择主张逾期利息、罚息或者其他费用，也可以一并主张，但总计超过年利率24%的部分，人民法院不予支持&rdquo;的规定，本案中，原告杨晓超主张按月息2%支付利息、罚息、违约金，未超出法律限制性规定，本院予以支持。被告王腾经传票传唤，无正当理由拒不到庭参加诉讼，视为自动放弃诉讼权利，应予缺席判决。"
#         }
#     ]
# }

# insert_data_to_mysql(anyou_name=test_data["anyou_name"],
#                      source=test_data["source"],
#                      data=test_data["insert_data"])
