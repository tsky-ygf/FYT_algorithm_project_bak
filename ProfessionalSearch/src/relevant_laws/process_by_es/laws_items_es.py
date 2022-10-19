#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 09:32
# @Author  :
# @Site    :
# @File    : laws_items_es.py
# @Software: PyCharm
import re
import json

import pandas as pd
from elasticsearch import Elasticsearch

from BasicTask.SearchEngine.es_tools import BaseESTool, helpers
import connectorx as cx
from loguru import logger

case_es_tools = {
    "host": "172.19.82.227",
    "user": "root",
    "passwords": "Nblh@2022",
    "db_name": "falvfagui_data",
    "table_list": [
        "flfg_result_dfxfg",
        "flfg_result_falv",
        "flfg_result_sfjs",
        "flfg_result_xf",
        "flfg_result_xzfg",
        "flfg_result_jcfg",
    ],
    "index_name": "flfg",
    "debug": False,
}


# es_hosts = "127.0.0.1:9200"


class LawItemsESTool(BaseESTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.valid_mapping = {"有效": 0, "": 1, "已修改": 2, "尚未生效": 3, "已废止": 4}
        self.legal_mapping = {
            "法律": 0,
            "行政法规": 1,
            "监察法规": 2,
            "司法解释": 3,
            "宪法": 5,
            "地方性法规": 4,
        }

        with open(
                "ProfessionalSearch/config/relevant_laws/minshi_item.json"
        ) as f:
            self.minshi_item = json.load(f)

        with open(
                "ProfessionalSearch/config/relevant_laws/xingshi_item.json"
        ) as f:
            self.xingshi_item = json.load(f)

    @staticmethod
    def chuli_title(old_s):  # 保留中文、大小写、数字
        cop = re.compile("[^\u4e00-\u9fa5^0-9]")  # 匹配不是中文、大小写、数字的其他字符
        nwe_s = cop.sub("", old_s)  # 将old_s中匹配到的字符替换成空s字符
        return nwe_s

    def get_query(self, table_name, start, end, way):
        if self.debug:
            query = "select * from {} limit 1000".format(table_name)
        else:
            if way == 'insert':
                query = "select id,isValid,resultChapter,resultClause,resultSection,title,md5Clause,source,prov from {}".format(
                    table_name
                )
            elif way == 'update':
                query = "select id,isValid,resultChapter,resultClause,resultSection,title,md5Clause,source,prov,update_time from {} where DATE_SUB(CURDATE(),INTERVAL 2 DAY) <= DATE(update_time)".format(
                    table_name
                )
            elif way == 'create':
                query = "select id,isValid,resultChapter,resultClause,resultSection,title,md5Clause,source,prov from {}".format(
                    table_name
                )
        logger.info(query)
        return query

    def get_df_data_from_db(self, table_name, start, end, way):
        print("sql查询开始")
        query = self.get_query(table_name, start, end, way)

        if self.use_big_data:
            return_type = "dask"
        else:
            return_type = "pandas"
        res_df = cx.read_sql(
            self.mysql_url, query=query, return_type=return_type, partition_num=10
        )
        print("sql查询完成")
        return res_df

    def __call__(self, way):
        if way == 'insert':
            self.es_init()
        for table_name in self.table_list:
            if way == 'insert':
                print("table_name", table_name)
                df_data = self.get_df_data_from_db(table_name, 0, 0, way)
                self.insert_data_to_es(df_data)
                logger.info("insert data to es success from {}".format(table_name))
            elif way == 'update':
                print("table_name", table_name)
                df_data = self.get_df_data_from_db(table_name, 0, 0, way)
                self.update_data_from_es_parall(6, 1000, df_data, table_name)
                logger.info("update data to es success from {}".format(table_name))
            elif way == 'create':
                print("table_name", table_name)
                df_data = self.get_df_data_from_db(table_name, 0, 0, way)
                self.update_data_from_es_parall(6, 1000, df_data, table_name)
                logger.info("update data to es success from {}".format(table_name))

    def handle_es(self, df_data):
        # 构造迭代器
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                "id",
                "isValid",
                "resultChapter",
                "resultClause",
                "resultSection",
                "title",
                "md5Clause",
                "source",
                "prov",
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            data_body["isValid_weight"] = self.valid_mapping[data_body["isValid"]]
            data_body["legal_type_weight"] = self.legal_mapping[data_body["source"]]
            title = self.chuli_title(data_body["title"])
            data_body["title_weight"] = 0
            if title in self.minshi_item:
                data_body["title_weight"] += self.minshi_item[title]
            if title in self.xingshi_item:
                data_body["title_weight"] += self.xingshi_item[title]
            yield {
                "_index": self.index_name,
                "_type": "_doc",
                "_id": row["md5Clause"],
                "_source": data_body,
            }

    def update_es_parall(self, df_data, table_name):
        count = 0
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                "id",
                "isValid",
                "resultChapter",
                "resultClause",
                "resultSection",
                "title",
                "md5Clause",
                "source",
                "prov",
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            data_body["isValid_weight"] = self.valid_mapping[data_body["isValid"]]
            data_body["legal_type_weight"] = self.legal_mapping[data_body["source"]]
            title = self.chuli_title(data_body["title"])
            data_body["title_weight"] = 0
            if data_body["resultChapter"]:
                data_body["resultChapter"] = bytes.decode(data_body["resultChapter"])
            else:
                data_body["resultChapter"] = ""
            if data_body["resultClause"]:
                data_body["resultClause"] = bytes.decode(data_body["resultClause"])
            else:
                data_body["resultClause"] = ""
            if data_body["resultSection"]:
                data_body["resultSection"] = bytes.decode(data_body["resultSection"])
            else:
                data_body["resultSection"] = ""

            if title in self.minshi_item:
                data_body["title_weight"] += self.minshi_item[title]
            if title in self.xingshi_item:
                data_body["title_weight"] += self.xingshi_item[title]

            if count < 30:
                print("doc_id:", data_body["id"], ";resultChapter:", data_body["resultChapter"])
                print("title:", data_body["title"])
                print("source:", data_body["source"])
                print("prov:", data_body["prov"])
                print("==============================")
            if count % 10000 == 0:
                print(count)
            count = count + 1


            yield {
                "_op_type": "update",
                "_index": self.index_name,
                "_type": "_doc",
                "_id": row["md5Clause"],
                "doc": data_body,
            }


def process_law_to_es(way):
    laws_es = LawItemsESTool(**case_es_tools)
    laws_es(way)

    query_dict = {
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {"match_phrase": {"resultClause": "涉外"}},
                                {"match_phrase": {"title": "涉外"}},
                            ]
                        }
                    },
                    {
                        "bool": {
                            "should": [
                                {"match_phrase": {"resultClause": "离婚"}},
                                {"match_phrase": {"title": "离婚"}},
                            ]
                        }
                    },
                    {"terms": {"isValid.keyword": ["有效"]}},
                    {"terms": {"source.keyword": ["法律"]}},
                ]
            }
        },
        "size": 10,
        "sort": [
            {"title_weight": {"order": "desc"}},
            {"isValid_weight": {"order": "asc"}},
            {"legal_type_weight": {"order": "asc"}},
        ],
    }
    res = laws_es.search_data_from_es(query_body=query_dict)

    print(res)
