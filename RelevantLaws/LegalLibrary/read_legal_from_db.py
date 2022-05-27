#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/26 14:17
# @Author  : Adolf
# @Site    : 
# @File    : read_legal_from_db.py
# @Software: PyCharm
import pymysql
import pandas as pd
from pprint import pprint
from elasticsearch import Elasticsearch

pd.set_option('display.max_columns', None)

connect_big_data = pymysql.connect(host='172.19.82.227',
                                   user='root', password='Nblh@2022',
                                   db='big_data_ceshi227')

sqlcmd = """select * from test_falvfagui_data.test_flfg_result_xf"""

data = pd.read_sql(sqlcmd, connect_big_data)
# print(data.head())

es = Elasticsearch('172.19.82.199', port=9200)

# print(es.cat.indices())
es.indices.delete(index='fvfg', ignore=[400, 404])
result = es.indices.create(index='fvfg', ignore=400)
# print(result)
for index, row in data.iterrows():
    # pprint(row.to_dict())
    data_ori = row.to_dict()

    use_data = ['isValid', 'resultChapter', 'resultClause', 'resultSection','title']
    data_body = {key: value for key, value in data_ori.items() if key in use_data}

    # pprint(data_body)
    # exit()
    response = es.index(index='fvfg', doc_type=data_ori["source"], id=data_ori["md5Clause"],body=data_body)
    # break

result = es.search(index='fvfg', doc_type='宪法')
pprint(result)