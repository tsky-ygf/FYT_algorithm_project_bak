#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/2 21:04
# @Author  : Adolf
# @Site    : 
# @File    : es_tools.py
# @Software: PyCharm
import pandas as pd
import connectorx as cx
from elasticsearch import Elasticsearch, helpers


class BaseESTool:
    def __init__(self,
                 host='172.19.82.227',
                 port='3306',
                 user='root',
                 passwords='Nblh@2022',
                 db_name='falvfagui_data',
                 es_host='127.0.0.1:9200',
                 index_name='flfg',
                 table_list=["flfg_result_dfxfg"],
                 debug=False,
                 use_big_data=False):
        self.mysql_url = f"mysql://{user}:{passwords}@{host}:{port}/{db_name}"

        self.es_host = es_host
        self.index_name = index_name

        self.db_name = db_name
        self.table_list = table_list

        self.debug = debug
        self.use_big_data = use_big_data

    # 从数据库中获取数据
    def get_df_data_from_db(self, table_name):
        if self.debug:
            query = "select * from {} limit 1000".format(table_name)
        else:
            query = "select * from {}".format(table_name)

        if self.use_big_data:
            return_type = "dask"
        else:
            return_type = "pandas"
        res_df = cx.read_sql(self.mysql_url,
                             query=query,
                             return_type=return_type,
                             partition_num=10)
        return res_df

    # 初始化es
    def es_init(self):
        es = Elasticsearch(hosts=self.es_host)
        # 重新创建索引
        es.indices.delete(index=self.index_name, ignore=[400, 404])
        es.indices.create(index=self.index_name, ignore=400)

    # 处理数据
    def handle_es(self, df_data, *args, **kwargs):
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            yield {"_index": self.index_name, "_type": "_doc", "_source": data_ori}

    # 插入数据到es
    def insert_data_to_es(self, *args, **kwargs):
        es = Elasticsearch(hosts=self.es_host)

        # for table_name in self.table_list:
        #     df_data = self.get_df_data_from_db(table_name=table_name)
        # 插入数据
        helpers.bulk(es, self.handle_es(*args, **kwargs))

    # 从es中搜索数据
    def search_data_from_es(self, query_body):
        # 查询数据
        es = Elasticsearch(hosts=self.es_host)
        res = es.search(index=self.index_name, body=query_body)
        res_list = [hit['_source'] for hit in res['hits']['hits']]
        df = pd.DataFrame(res_list)
        df.fillna('', inplace=True)
        return df

    def delete_data_from_es(self, query_body):
        es = Elasticsearch(hosts=self.es_host)
        res = es.delete_by_query(index=self.index_name, body=query_body)
        print(res)

    def __call__(self):
        self.es_init()

        for table_name in self.table_list:
            df_data = self.get_df_data_from_db(table_name)
            self.insert_data_to_es(df_data, table_name)
