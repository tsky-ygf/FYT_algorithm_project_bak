#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 16:54
# @Author  : Adolf
# @Site    : 
# @File    : tmp.py
# @Software: PyCharm
# from datetime import datetime
# from elasticsearch import Elasticsearch
import jieba

text = '名誉侵权'

res1 = ' '.join(jieba.cut(text))
print(res1)
# es = Elasticsearch()
#
# doc = {
#     'author': 'kimchy',
#     'text': 'Elasticsearch: cool. bonsai cool.',
#     'timestamp': datetime.now(),
# }
# res = es.index(index="test-index", id=1, body=doc)
# print(res['result'])
#
# res = es.get(index="test-index", id=1)
# print(res['_source'])
#
# # es.indices.refresh(index="test-index")
#
# res = es.search(index="test-index", body={"query": {"match_all": {}}})
# print("Got %d Hits:" % res['hits']['total']['value'])
# for hit in res['hits']['hits']:
#     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])

