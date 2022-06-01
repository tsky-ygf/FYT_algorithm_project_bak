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
from elasticsearch import Elasticsearch, helpers

pd.set_option('display.max_columns', None)


def connect_mysql():
    # 连接mysql
    connect_big_data = pymysql.connect(host='172.19.82.227',
                                       user='root', password='Nblh@2022',
                                       db='big_data_ceshi227')
    return connect_big_data


index_name = "fvfg"

db_name = 'test_falvfagui_data'
table_name_list = ['test_flfg_result_xf', 'test_flfg_result_xzfg', 'test_flfg_result_falv']


# sqlcmd = """select * from test_falvfagui_data.test_flfg_result_xf"""
# sqlcmd = """select * from test_falvfagui_data.test_flfg_result_xzfg"""
# sqlcmd = """select * from test_falvfagui_data.test_flfg_result_falv"""

def get_df_from_sql(table_name):
    connect_big_data = connect_mysql()
    # 获取数据
    sqlcmd = """select * from {}.{}""".format(db_name, table_name)
    df = pd.read_sql(sqlcmd, connect_big_data)
    return df


def es_init():
    es.indices.delete(index=index_name, ignore=[400, 404])
    es.indices.create(index=index_name, ignore=400)


# 构造es插入迭代器
def handle_es(df_data):
    # 构造迭代器
    for index, row in df_data.iterrows():
        data_ori = row.to_dict()
        use_data = ['isValid', 'resultChapter', 'resultClause', 'resultSection', 'title', 'md5Clause', 'source']
        data_body = {key: value for key, value in data_ori.items() if key in use_data}
        yield {"_index": index_name, "_type": "_doc", "_source": data_body}


#
# inset_data
def insert_data_to_es():
    es = Elasticsearch(hosts="127.0.0.1:9200")

    for table_name in table_name_list:
        df_data = get_df_from_sql(table_name)
        # 插入数据
        helpers.bulk(es, handle_es(df_data))


#

def search_data_from_es(query_body, _index_name='fvfg'):
    # 查询数据
    es = Elasticsearch(hosts="127.0.0.1:9200")
    res = es.search(index=_index_name, body=query_body)
    # print("Got %d Hits:" % res['hits']['total']['value'])
    res_list = [hit['_source'] for hit in res['hits']['hits']]
    df = pd.DataFrame(res_list)
    df.fillna('', inplace=True)
    # pprint(res_list[0])
    return df
    # for index, hit in enumerate(res['hits']['hits']):
    #     print(index)
    #     print(hit["_source"])

    # return res


if __name__ == '__main__':
    query_dict = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"resultClause": "人民"}},
                    {'term': {'isValid.keyword': '有效'}},
                    {"term": {"source.keyword": "宪法"}}, ],
            }
        },
        "size": 10,
    }
    # print(get_df_from_sql(table_name_list[0]))
    # es_init()
    # insert_data_to_es()
    pd.set_option('display.width', 1000)
    # print(search_data_from_es({"query": {"match_all": {}}, "size": 10}))
    print(search_data_from_es(query_dict))
    # print(es.cat.indices())
