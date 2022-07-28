#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 09:15
# @Author  : Adolf
# @Site    : 
# @File    : entity_annotation.py
# @Software: PyCharm
import logging

import pymysql
# from plogger.info import plogger.info
import pandas as pd
from pathlib import Path
import requests
import traceback
import datetime
from loguru import logger


pd.set_option('display.max_columns', None)

__all__ = ['get_anyou_list', 'get_case_feature_dict', 'get_base_data_dict', 'get_base_annotation_dict',
           'insert_data_to_mysql','get_second_check','get_day_work_count','get_source_content',
           'check_username','get_login_password','save_username_password','save_second_check_proson']

connect_big_data = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='big_data_ceshi227')
connect_labels_marking_records = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='labels_marking_records')
connect_big_data_ceshi = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='big_data_ceshi227')
connect_login_uers_data = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='login_user_data')

def get_anyou_list():
    anyou_list = []
    csv_config_path = Path("data/LawsuitPrejudgment/CaseFeatureConfig/")
    for csv_path in csv_config_path.glob("**/*.csv"):
        anyou_list.append(csv_path.name.replace('.csv', ''))
    return anyou_list


# logger.info(get_anyou_list())
def get_case_feature_dict(anyou_name):
    # logger.info(anyou_name)
    anyou_case_feature_dict = {}
    df = pd.read_csv(Path("data/LawsuitPrejudgment/CaseFeatureConfig/") / (anyou_name + ".csv"))
    for index, row in df.iterrows():
        # logger.info(row['case'],row['feature'])
        if row['case'] not in anyou_case_feature_dict:
            anyou_case_feature_dict[row['case']] = []
        anyou_case_feature_dict[row['case']].append(row['feature'])
    return anyou_case_feature_dict

# logger.info(get_case_feature(anyou_name="借贷纠纷_民间借贷"))


# def read_data_from_mysql():
#     sql_con = '''
#             select f2,f13,f40,f44 from big_data_ceshi227.case_list_original_hetong
#             WHERE f12="民间借贷纠纷" AND f10="判决" AND (LENGTH(f40)>1) limit 100;
#            '''
#
#     data = pd.read_sql(sql_con, con=connect_big_data)
#     # connect_big_data.close()
#     return data

def get_base_data_dict(anyou_name):
    cursor = connect_big_data.cursor()
    anyou_type, anyou_x = anyou_name.split('_')
    if anyou_name == "借贷纠纷_民间借贷":
        sql_con = '''
        select f2,f13,f40,f44 from labels_marking_records.case_list_original_hetong 
        WHERE f12="民间借贷纠纷" AND f10="判决" AND (LENGTH(f40)>1) AND (f50 is NULL) limit 1;
        '''
    elif anyou_name == "婚姻继承_离婚":
        sql_con = '''
        select f2,f13,f40,f44 from labels_marking_records.case_list_original_hunyinjiating 
        WHERE f12="离婚纠纷" AND f10="判决" AND (LENGTH(f40)>1) AND (f50 is NULL) limit 1;
        '''
    else:
        raise Exception("暂时不支持该案由")

    data = pd.read_sql(sql_con, con=connect_big_data)
    base_data_dict = {
        "case_id": data["f2"].values[0],
        "data": [
            {"name": "原告诉称", "content": data["f40"].values[0]},
            {"name": "本院查明", "content": data["f44"].values[0]},
            {"name": "本院认为", "content": data["f13"].values[0]}]
    }

    if anyou_name == "借贷纠纷_民间借贷":
        sql_update_con = '''
            UPDATE labels_marking_records.case_list_original_hetong SET f50='{}' WHERE f2='{}';
        '''.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data["f2"].values[0])
    if anyou_name == "婚姻继承_离婚":
        sql_update_con = '''
            UPDATE labels_marking_records.case_list_original_hunyinjiating SET f50='{}' WHERE f2='{}';
        '''.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data["f2"].values[0])

    try:
        cursor.execute(sql_update_con)
        connect_big_data.commit()
    except:
        connect_big_data.rollback()
        raise RuntimeError("更新数据库时间失败")

    cursor.close()
    return base_data_dict

def get_source_content(key):
    cursor = connect_big_data_ceshi.cursor(cursor=pymysql.cursors.DictCursor)
    sql = f"""
        select f13
        from case_list_original_base_info
        where f2 = '{key}'
        """
    try:
        cursor.execute(sql)
        res = cursor.fetchall()
    except:
        connect_big_data.rollback()
        raise RuntimeError("查询数据库时间失败")
    cursor.close()
    return res

def get_base_annotation_dict(anyou_name, sentence):
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

# logger.info(get_base_annotation_dict(anyou_name="借贷纠纷_民间借贷",
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
    # logger.info(labelingdate)
    factor = "###".join(factor)

    sql_con = """
        INSERT INTO labels_marking_records.labels_law_entity_feature (id,suqiu,jiufen_type,source,content,mention,situation,factor,
        startposition,endpostion,pos_neg,labelingdate,labelingperson,checkperson) VALUES ('{}','{}','{}','{}','{}','{}','{}','{}',
        {},{},{},'{}','{}',NULL); 
    """.format(id, suqiu, jiufen_type, source, content, mention, situation, factor, start_pos, end_pos, pos_or_neg,
               labelingdate, labelingperson)

    # logger.info(sql_con)

    try:
        # 执行sql语句
        cursor.execute(sql_con)
        # 执行sql语句
        connect_big_data.commit()
    except:
        # 发生错误时回滚
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

def get_base_data_dict(anyou_name):
    print(anyou_name)
    pass

get_base_data_dict("借贷纠纷_民间借贷")

def save_username_password(username,password):
    # 保存 用户 账号密码
    # return: 正常返回 1
    try:
        cur = connect_login_uers_data.cursor(cursor=pymysql.cursors.DictCursor)
        sql = f"""insert into user_data (username,password) value ('{username}','{password}')"""
        res = cur.execute(sql)
        connect_login_uers_data.commit()
        cur.close()
        # connect_login_uers_data.close()
        return res
    except pymysql.err.OperationalError:
        logger.info(traceback.format_exc())
        logger.error(f"链接数据库 login_user_data 异常，查看数据库链接信息是否有问题")
        raise ConnectionError("method_name:check_username, 连接数据库 login_user_data 异常，查看数据库链接信息是否有问题")
    except Exception as e:
        logger.info(traceback.format_exc())
        connect_big_data.rollback()
        raise RuntimeError("查询数据库时间失败")

def get_dict_value(data_dict,key):
    value = data_dict.get(key)
    if value:
        if isinstance(value, str):
            return f"\'{value}\'"
        elif isinstance(value,int):
            return f"{value}"
        elif isinstance(value,datetime.datetime):
            return f"\'{value}\'"
        else:
            return value
    else:
        return 'NULL'

def get_day_work_count(person):
    # 当天时间 eg: 2022-05-18 00:00:00
    this_day = datetime.datetime.now().strftime('%Y-%m-%d')
    this_day = datetime.datetime.strptime(this_day, '%Y-%m-%d')
    # 明天时间 eg: 2022-05-19 00:00:00
    next_day = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    next_day = datetime.datetime.strptime(next_day, '%Y-%m-%d')
    cursor = connect_labels_marking_records.cursor(cursor=pymysql.cursors.DictCursor)
    sql = f"""
    select labelingperson,count(labelingperson) as num 
    from labels_law_entity_feature
    WHERE labelingperson ='{person}'
    and labelingdate between '{this_day}' and '{next_day}' 
    group by labelingperson;
    """
    try:
        cursor.execute(sql)
        res = cursor.fetchall()
        # logger.info(f"res:{res}")
    except:
        connect_big_data.rollback()
        raise RuntimeError("查询数据库时间失败")
    cursor.close()
    return res

def get_second_check(anyou):
    if anyou == "借贷纠纷_民间借贷":
        try:
            cursor = connect_labels_marking_records.cursor(cursor=pymysql.cursors.DictCursor)
            sql = f"""
            SELECT * 
            FROM labels_law_entity_feature 
            WHERE suqiu='借贷纠纷' 
            and jiufen_type='民间借贷' 
            and checkperson is null 
            order by id
            limit 1;
            """
            cursor.execute(sql)
            res = cursor.fetchall()
        except Exception as e:
            connect_big_data.rollback()
            raise RuntimeError("查询数据库时间失败")
        cursor.close()
        return res
    return ''

def save_second_check_proson(data_dict):
    cursor = connect_labels_marking_records.cursor()
    sql_update_con = f'''
                        UPDATE labels_law_entity_feature
                        SET
                            suqiu={get_dict_value(data_dict, 'suqiu')},
                            jiufen_type={get_dict_value(data_dict, 'jiufen_type')},
                            source={get_dict_value(data_dict, 'source')},
                            content={get_dict_value(data_dict, 'content')},
                            mention={get_dict_value(data_dict, 'mention')},
                            situation={get_dict_value(data_dict, 'situation')},
                            factor={get_dict_value(data_dict, 'factor')},
                            startposition={get_dict_value(data_dict, 'startposition')},
                            endpostion={get_dict_value(data_dict, 'endpostion')},
                            pos_neg={get_dict_value(data_dict, 'pos_neg')},
                            labelingdate={get_dict_value(data_dict, 'labelingdate')},
                            labelingperson={get_dict_value(data_dict, 'labelingperson')},
                            checkperson={get_dict_value(data_dict, 'checkperson')},
                            checkResult={get_dict_value(data_dict, 'checkResult')}
                         WHERE id={get_dict_value(data_dict, 'id')};
                    '''
    try:
        cursor.execute(sql_update_con)
        connect_labels_marking_records.commit()
        cursor.close()
    except:
        logger.error(traceback.format_exc())
        connect_labels_marking_records.rollback()
        raise RuntimeError("更新数据库时间失败")

if __name__ == '__main__':
    pass
#     print(get_case_feature_dict1('婚姻继承_离婚'))
    # print(get_second_check('借贷纠纷_民间借贷'))
    # def get_second_check(anyou):
    #     anyou_type, anyou_x = anyou.split('_')
    #     if anyou == "借贷纠纷_民间借贷":

