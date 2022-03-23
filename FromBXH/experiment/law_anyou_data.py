#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/22 14:15
# @Author  : Adolf
# @Site    : 
# @File    : law_anyou_data.py
# @Software: PyCharm
import pymysql
import pandas as pd


def get_suqiu_anyou_dict():
    # 1. 连接数据库
    sql = '''
    select distinct problem,suqiu,anyou,suqiu_desc FROM justice.algo_train_law_case_y_keyword where status=
    '''
    connect = pymysql.connect(host="rm-bp18g0150979o8v4tlo.mysql.rds.aliyuncs.com", user="justice_user_03",
                              password="justice_user_03_pd_!@#$", db="justice")
    data = pd.read_sql(sql, con=connect)

    # 2.获取需要的数据
    suqiu_anyou_dict = data['anyou'].groupby([data['problem'], data['suqiu']]).agg(lambda x: list(set(x)))
    # groupby(df_keyword['factor_id'])：相当于sql中的group by factor_id；agg(lambda x: list(x)[0])将factor_name变成一个list,并取第一个
    print("suqiu_anyou_dict:", suqiu_anyou_dict)

    print("suqiu_anyou_dict['婚姻家庭']['财产分割']:", suqiu_anyou_dict['婚姻家庭']['财产分割'])
    # 3. 关闭数据库连接
    connect.close()
    return suqiu_anyou_dict


suqiu_anyou_dict = get_suqiu_anyou_dict()
