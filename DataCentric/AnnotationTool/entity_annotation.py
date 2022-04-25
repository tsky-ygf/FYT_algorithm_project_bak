#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 09:15
# @Author  : Adolf
# @Site    : 
# @File    : entity_annotation.py
# @Software: PyCharm
import pymysql
import pandas as pd

connect_big_data = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='big_data_ceshi227')

sql_con = '''
    select f7,f10,f12,f13,f14,f30,f40,f42,f44 from big_data_ceshi227.case_list_original_hetong WHERE f12="民间借贷纠纷" and f10="判决"
       '''
data = pd.read_sql(sql_con, con=connect_big_data)
# connect_big_data.close()


