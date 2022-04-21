# -*- coding: utf-8 -*-
import pymysql
import datetime
import pandas as pd
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../common'))
from data_util import *
from data_analyze import *

# ES安装：https://www.elastic.co/cn/
# 安装kibana
# 启动：进入es安装目录，bin/elasticsearch；进入kibana安装目录，bin/kibana


es = Elasticsearch()
LAW_ES_INDEX_NAME = 'law_content_search'
CASE_ES_INDEX_NAME = 'case_content_search'


##############################################################################################################################################
#
# 法条检索ES数据库
#
##############################################################################################################################################

def get_law_data(from_database=True):
    if from_database:
        connect_big_data = pymysql.connect(host='192.168.1.253', port=3366,
                                           user='justice_user_03', password='justice_user_03_pd_!@#$',
                                           db='justice_big_data')
        sql1 = '''
            select publish_date, law_name, chapter, clause, content, property, province from justice_big_data.law_dictionary_npc
            where province = ''
        '''
        data_law = pd.read_sql(sql1, con=connect_big_data)
        print("从数据库的表取出了:", len(data_law))
        connect_big_data.close()

        data_law['law_name'] = data_law['law_name'].apply(lambda x: x.replace('《', '').replace('》', ''))
        data_law['publish_date'] = data_law['publish_date'].apply(lambda x: x.replace('年', '-').replace('月', '-').replace('日', ''))
        data_law['content_simple'] = data_law['content'].apply(pos_cut)
        data_law['fatiao'] = (data_law['law_name']+data_law['clause']).apply(chinese_filter)

        data_case = pd.read_csv('../data/new_case.csv', encoding='utf-8')
        data_case = data_case.fillna('')
        data_case = data_case[data_case['fatiao'].str.len()>0]
        data_case = data_case.reset_index().drop('index', axis=1)
        temp = data_case['fatiao'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('fatiao')
        data_case = data_case.drop('fatiao', axis=1).join(temp)
        data_case['fatiao'] = data_case['fatiao'].apply(chinese_filter)
        fs_dict = data_case['serial'].groupby(data_case['fatiao']).agg(lambda x: list(x))

        data_law['serials'] = data_law['fatiao'].apply(lambda x: '|'.join(fs_dict[x][:20]) if x in fs_dict else '')
        data_law.to_csv('../data/law.csv', index=False, encoding='utf-8')
    else:
        data_law = pd.read_csv('../data/law.csv', encoding='utf-8')
        data_law = data_law.fillna('')
    return data_law


# 1.创建数据库
def create_law_es_indices():

    # 移除已经存在的特定索引
    if es.indices.exists(index=LAW_ES_INDEX_NAME):
        print('remove index')
        es.indices.delete(index=LAW_ES_INDEX_NAME)

    es.indices.create(index=LAW_ES_INDEX_NAME, body={
        "mappings": {
            "_doc": {
                "properties": {
                    "index": {
                        "type": "integer"
                    },
                    "law_name": {
                        "type": "text"
                    },
                    "chapter": {
                        "type": "text"
                    },
                    "clause": {
                        "type": "text"
                    },
                    "content": {
                        "type": "text"
                    },
                    "content_simple": {
                        "type": "text"
                    },
                    "property": {
                        "type": "text"
                    },
                    "province": {
                        "type": "text"
                    },
                    "publish_date": {
                        "type": "date"
                    },
                    "serials":{
                        "type": "text"
                    }
                }
            }
        }
    })


# 2.插入数据到特定的索引库中
def add_law_docs(data_law):
    for index, row in data_law.iterrows():
        doc = {'index': index,
               'law_name': row['law_name'],
               'chapter': row['chapter'],
               'clause': row['clause'],
               'content': row['content'],
               'content_simple': row['content_simple'],
               'property': row['property'],
               'province': row['province'],
               'publish_date': row['publish_date'],
               'serials': row['serials']}
        if index % 10000 == 0:
            print("index:", index)

        yield {
            '_index': LAW_ES_INDEX_NAME,
            '_type': '_doc',
            '_source': doc,
        }


# 批量插入
def add_law_docs_bulk(data_law):
    bulk(es, add_law_docs(data_law))


# 3.使用搜索的例子
def search_law_example():
    law_name = '婚姻法'
    body = {
        "size": 10,
        "query": {
            "multi_match": {
                "query": law_name,
                "fields": ["law_name"],
                "type": "phrase",
                "slop": 3
            }
        },
        "_source": ['law_name', 'chapter', 'clause', 'content'],
        "sort": [
            {"index": {"order": "asc"}}
        ]
    }
    res = es.search(index=LAW_ES_INDEX_NAME, body=body)  # res = es.search(index="test-index", body={"query": {"match_all": {}}})
    print("Got %d Hits:" % len(res['hits']['hits']))
    for hit in res['hits']['hits']:
        print(hit['_source'])


def law_test():
    data_law = get_law_data()
    print('data size: %s' % (len(data_law)))
    create_law_es_indices()
    add_law_docs_bulk(data_law)
    # search_law_example()



##############################################################################################################################################
#
# 案例检索ES数据库
#
##############################################################################################################################################


def get_case_data(from_database=True):
    def _date_check(x):
        try:
            datetime.datetime.strptime(x, '%Y-%m-%d')
            return True
        except:
            return False

    if from_database:
        database = ['case_list_original_hetong', 'case_list_original_hunyinjiating', 'case_list_original_labor',
                    'case_list_original_labor_2', 'case_list_original_qinquan', 'case_list_original_rengequan',
                    'case_list_original_wuquan', 'case_list_original_zhengquan', 'case_list_original_zhishichanquan']

        data_case = []
        connect_big_data = pymysql.connect(host='192.168.1.253', port=3366,
                                           user='justice_user_03', password='justice_user_03_pd_!@#$',
                                           db='justice_big_data')
        for db in database:
            sql1 = '''
                select f1 as title, f3 as province, f5 as serial, f12 as anyou, f14 as date, f41 as court, f7 as raw_content
                from justice_big_data.%s where f14 > '2018'
            ''' % (db)
            temp = pd.read_sql(sql1, con=connect_big_data)
            print("数据库：", db, "从数据库的表取出了:", len(temp))
            data_case.append(temp)
        connect_big_data.close()

        data_case = pd.concat(data_case, sort=False)
        data_case = data_case.reset_index().drop('index', axis=1)

        data_case = data_case[data_case['date'].apply(_date_check)]
        data_case['content'] = data_case['raw_content'].apply(html_clean)
        data_case['sucheng'] = data_case['raw_content'].apply(sucheng_extract)
        data_case['suqing'] = data_case['sucheng'].apply(lambda x: x[1] if x is not None else None)
        data_case['sucheng'] = data_case['sucheng'].apply(lambda x: x[0] if x is not None else None)
        data_case['fatiao'] = data_case['raw_content'].apply(fatiao_extract)
        data_case['fatiao'] = data_case['fatiao'].apply(fatiao_correct)
        data_case['panjue'] = data_case['raw_content'].apply(panjue_extract)
        data_case['panjue'] = data_case['panjue'].apply(lambda x: x['判决'] if x is not None and '判决' in x else None)
        data_case = data_case[['title', 'province', 'serial', 'anyou', 'date', 'court', 'sucheng', 'suqing', 'fatiao', 'panjue', 'content']]
        data_case.to_csv('../data/case.csv', index=False, encoding='utf-8')
    data_case = pd.read_csv('../data/case.csv', encoding='utf-8')
    data_case = data_case.dropna()
    data_case = data_case.reset_index().drop('index', axis=1)
    print('data size: %s' % (len(data_case)))
    return data_case


def get_new_case_data(from_database=True):

    def _date_check(x):
        try:
            datetime.datetime.strptime(str(x), '%Y-%m-%d')
            return True
        except:
            return False

    def _content_process(x):
        if x.startswith('</p>'):
            x = x[4:]
        if not x.endswith('</p>'):
            x = x + '</p>'
        return x

    if from_database:
        connect_big_data = pymysql.connect(host='192.168.1.253', port=3366,
                                           user='justice_user_03', password='justice_user_03_pd_!@#$',
                                           db='justice_big_data')
        sql1 = '''
            select f28 as title, f5 as province, f3 as serial, f9 as anyou, f18 as date, f10 as court, f30, f31, f32, f33, f34, f35, f36
            from justice_big_data.case_new_civil
        '''
        data_case = pd.read_sql(sql1, con=connect_big_data)
        print("数据库：case_new_civil, 从数据库的表取出了:", len(data_case))
        connect_big_data.close()

        data_case = data_case.fillna('')
        data_case = data_case[data_case['date'].apply(_date_check)]
        data_case['anyou'] = data_case['anyou'].apply(lambda x: '|'.join(x.replace("['", '').replace("']", '').split("', '")))
        data_case['anyou'] = data_case['anyou'].str.replace('\[\]', '')
        data_case['raw_content'] = data_case['f30'] + data_case['f31'] + data_case['f32'] + data_case['f33'] + data_case['f34'] + data_case['f35'] + data_case['f36']
        data_case['raw_content'] = data_case['raw_content'].str.replace('#', '</p><p>')
        data_case['raw_content'] = data_case['raw_content'].apply(_content_process)
        data_case['content'] = data_case['raw_content'].apply(html_clean)
        data_case['sucheng'] = data_case['raw_content'].apply(sucheng_extract)
        data_case['sucheng'] = data_case['sucheng'].apply(lambda x: x[0] if x is not None else None)
        data_case['chaming'] = data_case['raw_content'].apply(chaming_extract)
        data_case['renwei'] = data_case['raw_content'].apply(renwei_extract)
        data_case['fatiao'] = data_case['raw_content'].apply(fatiao_extract)
        data_case['fatiao'] = data_case['fatiao'].apply(fatiao_correct)
        data_case['panjue'] = data_case['raw_content'].apply(panjue_extract)
        data_case['panjue'] = data_case['panjue'].apply(lambda x: x['判决'] if x is not None and '判决' in x else None)
        data_case = data_case[['title', 'province', 'serial', 'anyou', 'date', 'court', 'sucheng', 'chaming','renwei', 'fatiao', 'panjue', 'content']]
        data_case.to_csv('../data/new_case.csv', index=False, encoding='utf-8')

    data_case = pd.read_csv('../data/new_case.csv', encoding='utf-8')
    data_case = data_case.fillna('')
    data_case = data_case[data_case['fatiao'].str.len()>0]
    data_case = data_case.reset_index().drop('index', axis=1)
    print('data size: %s' % (len(data_case)))
    return data_case


# 1.创建数据库
def create_case_es_indices():

    # 移除已经存在的特定索引
    if es.indices.exists(index=CASE_ES_INDEX_NAME):
        print('remove index')
        es.indices.delete(index=CASE_ES_INDEX_NAME)

    es.indices.create(index=CASE_ES_INDEX_NAME, body={
        "mappings": {
            "_doc": {
                "properties": {
                    "serial": {
                        "type": "keyword",
                    },
                    "title": {
                        "type": "text",
                    },
                    "province": {
                        "type": "text",
                    },
                    "anyou": {
                        "type": "text",
                    },
                    "date": {
                        "type": "date",
                    },
                    "court": {
                        "type": "text",
                    },
                    "sucheng": {
                        "type": "text"
                    },
                    "chaming": {
                        "type": "text"
                    },
                    "renwei": {
                        "type": "text"
                    },
                    "fatiao": {
                        "type": "text"
                    },
                    "panjue": {
                        "type": "text"
                    },
                    "content": {
                        "type": "text"
                    }
                }
            }
        }
    })


# 2.插入数据到特定的索引库中
def add_case_docs(data_case):
    for index, row in data_case.iterrows():
        doc = {'serial': row['serial'],
               'title': row['title'],
               'province': row['province'],
               'anyou': row['anyou'],
               'date': row['date'],
               'court': row['court'],
               'sucheng': row['sucheng'],
               'chaming': row['chaming'],
               'renwei': row['renwei'],
               'fatiao': row['fatiao'],
               'panjue': row['panjue'],
               'content': row['content']}
        if index % 10000 == 0:
            print("index:", index)

        yield {
            '_index': CASE_ES_INDEX_NAME,
            '_type': '_doc',
            '_source': doc,
        }


# 批量插入
def add_case_docs_bulk(data_case):
    bulk(es, add_case_docs(data_case))


# 3.使用搜索的例子
def search_case_example():
    case_content = '车祸'
    body = {
        "size": 10,
        "query": {
            "multi_match": {
                "query": case_content,
                "fields": ["sucheng"],
                "type": "phrase",
                "slop": 3
            }
        },
        "_source": ['title', 'content'],
    }
    res = es.search(index=CASE_ES_INDEX_NAME, body=body)  # res = es.search(index="test-index", body={"query": {"match_all": {}}})
    print("Got %d Hits:" % len(res['hits']['hits']))
    for hit in res['hits']['hits']:
        print(hit['_source'])


def case_test():
    data_case = get_new_case_data()
    create_case_es_indices()
    add_case_docs_bulk(data_case)
    # search_law_example()


if __name__ == '__main__':
    case_test()
    law_test()
