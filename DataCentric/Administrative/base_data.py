#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 16:57
# @Author  : Adolf
# @Site    : 
# @File    : base_data.py
# @Software: PyCharm
import pandas as pd
import pymysql
from pathlib import Path
from pprint import pprint

# from sqlalchemy.engine import URL
# connection_string = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=172.19.82.227;DATABASE=big_data_ceshi227;UID=root;PWD=Nblh@2022"
# connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})

# from sqlalchemy import create_engine
# engine = create_engine(connection_url)

connect_big_data = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='big_data_ceshi227')

sqlcmd = """select * from test_falvfagui_data.xzcf_data limit 120"""
df = pd.read_sql(sqlcmd, connect_big_data)

# print(df)
#
for index, row in df.iterrows():
    content = row.to_dict()
    pprint(content)
    section = content['section']
    event_text = content['event_text']
#     # print(section)
#     # print(event_text)
#     file = Path('data/doccano_data/input_xz/') / 'xz_{}.txt'.format(index)
#     file.write_text('执法部门:{}\n行政处罚决定书:{}'.format(section, event_text))
    # break
