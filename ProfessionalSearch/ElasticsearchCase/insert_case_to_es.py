#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 09:53
# @Author  : Adolf
# @Site    : 
# @File    : insert_case_to_es.py
# @Software: PyCharm
from pprint import pprint
from BasicTask.SearchEngine.es_tools import BaseESTool

host = '172.19.82.227'
user = 'root'
passwords = 'Nblh@2022'
db = 'big_data_ceshi227'

db_name = "judgments_data"
# table_list = ["judgment_minshi_data", "judgment_xingshi_data"]
# table_list = ["judgment_xingshi_data"]
table_list = ["judgment_minshi_data"]

index_name = "case_index"
es_hosts = "127.0.0.1:9200"


# es = Elasticsearch(hosts=es_hosts)
# print(es.cat.indices())


# 为es新建一块分区
# es_init(_index_name=index_name, _es_hosts=es_hosts)

# df = get_df_from_sql(_db_name=db_name, _table_name=table_list[0])


def handle_es_v2(df_data, _db_name, _table_name):
    for index, row in df_data.iterrows():
        data_ori = row.to_dict()
        if index < 100:
            pprint(data_ori)
        use_data = ['uq_id', 'content', 'event_num', 'faYuan_name', 'jfType', 'event_type']
        data_body = {key: value for key, value in data_ori.items() if key in use_data}
        data_body['db_name'] = _db_name
        data_body['table_name'] = _table_name
        # print(data_body)
        # exit()
        yield {"_index": index_name, "_type": "_doc", "_source": data_body}


insert_data_to_es(_es_hosts=es_hosts,
                  _db_name=db_name,
                  _table_name_list=table_list,
                  _handle_es=handle_es_v2)

query_dict = {
    "query": {
        "match": {
            "content": "劫走"
        }
    }
}
res = search_data_from_es(query_body=query_dict,
                          _es_hosts=es_hosts,
                          _index_name=index_name)

print(res)
