#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 09:53
# @Author  : Adolf
# @Site    :
# @File    : insert_case_to_es.py
# @Software: PyCharm
from elasticsearch import Elasticsearch, helpers

from BasicTask.SearchEngine.es_tools import BaseESTool
import connectorx as cx
from loguru import logger

case_es_tools = {
    "host": "172.19.82.227",
    "user": "root",
    "passwords": "Nblh@2022",
    "db_name": "judgments_data",
    # "table_list": ["judgment_xingzheng_data", "judgment_xingshi_data", "judgment_zhixing_data", "judgment_minshi_data_cc"],
    "table_list": ["judgment_zhixing_data"],
    "index_name": "case_index_v2",
    "debug": False,
    "use_big_data": True,
}


# es_hosts = "127.0.0.1:9200"


class CaseESTool(BaseESTool):
    def get_query(self, table_name, start, end):
        if self.debug:
            query = "select * from {} limit 2000".format(table_name)
        else:
            query = "select id,uq_id,content,event_num,faYuan_name,jfType,event_type,province,wsTitle from {} limit {},{}".format(
                table_name, start, end
            )


        return query

    def get_df_data_from_db(self, table_name, start, end):
        print("sql查询开始")
        query = self.get_query(table_name, start, end)

        if self.use_big_data:
            return_type = "dask"
        else:
            return_type = "pandas"
        res_df = cx.read_sql(
            self.mysql_url, query=query, return_type=return_type, partition_num=10
        )
        print("sql查询完成")
        return res_df

    def create_data_to_es(self, *args, **kwargs):
        es = Elasticsearch(hosts=self.es_host)
        helpers.bulk(
            es,
            self.handle_es_create(*args, **kwargs),
            index=self.index_name,
            doc_type="doc",
            raise_on_exception=False,
            raise_on_error=False,
        )

    def handle_es_create(self, df_data, table_name):
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                "id",
                "uq_id",
                "content",
                "event_num",
                "faYuan_name",
                "jfType",
                "event_type",
                "province",
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            data_body["db_name"] = self.db_name
            data_body["table_name"] = table_name
            yield {
                "_op_type": "create",
                "_index": self.index_name,
                "_type": "_doc",
                "_id": row["uq_id"],
                "_source": data_body,
            }

    def handle_es_update(self, df_data, table_name):
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                "id",
                "uq_id",
                "content",
                "event_num",
                "faYuan_name",
                "jfType",
                "event_type",
                "province",
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            data_body["db_name"] = self.db_name
            data_body["table_name"] = table_name
            yield {
                "_op_type": "update",
                "_index": self.index_name,
                "_type": "_doc",
                "_id": row["uq_id"],
                "doc": data_body, # update 操作得用doc, index,create 操作用_source
            }

    def __call__(self):
        # self.es_init()
        start_end_mingshi = [[0, 2000000], [2000000, 2000000], [4000000, 2000000], [6000000, 2000000],[8000000, 2000000],[10000000, 2000000],[12000000, 1899519]] # 13899519   judgement_mingshi_data 0-16000000 插入的_id 为 id， 16000000-18972566 插入的_id 为uq_id, 除此之外，尽量插入用uq_id
        start_end_mingshi = [[12000000, 1899519]] # 13899519   judgement_mingshi_data 0-16000000 插入的_id 为 id， 16000000-18972566 插入的_id 为uq_id, 除此之外，尽量插入用uq_id

        # start_end_mingshi = [[0, 200]] # 13899519   judgement_mingshi_data 0-16000000 插入的_id 为 id， 16000000-18972566 插入的_id 为uq_id, 除此之外，尽量插入用uq_id

        start_end_xingshi = [[0, 2141203]] # 2141203
        start_end_xingzheng = [[0, 354424]] #354424
        start_end_zhixing = [[0, 2000000], [2000000, 2000000], [4000000, 3488256]] #  7488256
        # start_end_zhixing = [[0, 0]]  # 7488256
        for table_name in self.table_list:
            if table_name == 'judgment_xingzheng_data':
                start_end_df = pd.DataFrame(start_end_xingzheng)
            elif table_name == 'judgment_zhixing_data':
                start_end_df = pd.DataFrame(start_end_zhixing)
            elif table_name == 'judgment_minshi_data_cc':
                start_end_df = pd.DataFrame(start_end_mingshi)
            elif table_name == 'judgment_xingshi_data':
                start_end_df = pd.DataFrame(start_end_xingshi)
            for index, start_and_end in start_end_df.iterrows():
                print(table_name, start_and_end[0], start_and_end[1])
                df_data = self.get_df_data_from_db(table_name, start_and_end[0], start_and_end[1])
                if table_name == 'judgment_minshi_data_cc':
                    table_name == 'judgment_minshi_data'
                self.update_data_from_es_parall(6, 1000, df_data, table_name)
    def handle_es_parall(self, df_data, table_name):
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                "id",
                "uq_id",
                "content",
                "event_num",
                "faYuan_name",
                "jfType",
                "event_type",
                "province",
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            data_body["db_name"] = self.db_name
            data_body["table_name"] = table_name
            yield {
                "_index": self.index_name,
                "_id": row["uq_id"],
                "_type": "_doc",
                "_source": data_body,
            }

    def update_es_parall(self, df_data, table_name):
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                # "id",
                "uq_id",
                "content",
                "event_num",
                "faYuan_name",
                "jfType",
                "event_type",
                "province",
                "wsTitle"
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            data_body["db_name"] = self.db_name
            data_body["table_name"] = table_name
            yield {
                "_op_type": "update",
                "_index": self.index_name,
                "_type": "_doc",
                "_id": row["uq_id"],
                "doc": data_ori,
            }

    # def __call__(self):
    #     # self.es_init()
    #
    #     for table_name in self.table_list:
    #         df_data = self.get_df_data_from_db(table_name, 0, 0)
    #         self.update_data_from_es_parall(
    #             6, 600, df_data, "judgment_xingzheng_data_cc"
    #         )
    #
    #         logger.info("insert data to es success from {}".format(table_name))


if __name__ == "__main__":
    import pandas as pd

    pd.set_option("display.max_columns", None)

    case_es = CaseESTool(**case_es_tools)
    case_es()

    # query_dict = {"query": {"match": {"content": "买卖"}}}
    # res = case_es.search_data_from_es(query_body=query_dict)

    # print(res)
