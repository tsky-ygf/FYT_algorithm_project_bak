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

case_es_tools = {
    "host": "172.19.82.227",
    "user": "root",
    "passwords": "Nblh@2022",
    "db_name": "judgments_data",
    "table_list": ["judgment_minshi_data", "judgment_xingshi_data"],
    "index_name": "case_index",
    "debug": False,
    "use_big_data": True
}


# es_hosts = "127.0.0.1:9200"


class CaseESTool(BaseESTool):
    def get_query(self, table_name, start, end):
        if self.debug:
            query = "select * from {} limit 1000".format(table_name)
        else:
            query = "select id,uq_id,content,event_num,faYuan_name,jfType,event_type from {} limit {},{}".format(table_name, start, end)

        return query

    def get_df_data_from_db(self, table_name, start, end):
        print("sql查询开始")
        query = self.get_query(table_name, start, end)

        if self.use_big_data:
            return_type = "dask"
        else:
            return_type = "pandas"
        res_df = cx.read_sql(self.mysql_url,
                             query=query,
                             return_type=return_type,
                             partition_num=10)
        print("sql查询完成")
        return res_df

    def insert_data_to_es(self, *args, **kwargs):
        es = Elasticsearch(hosts=self.es_host)

        # for table_name in self.table_list:
        #     df_data = self.get_df_data_from_db(table_name=table_name)
        # 插入数据
        helpers.bulk(es, self.handle_es(*args, **kwargs), index='case_index', doc_type='doc'
                     , raise_on_exception=False, raise_on_error=False)

    def __call__(self):
        self.es_init()
        start_end_mingshi = [[0, 2000000], [2000000, 2000000], [4000000, 2000000], [6000000, 2000000],[8000000, 2000000],[10000000, 2000000],[12000000, 2000000]
                             ,[14000000, 2000000]]
        start_end_xingshi = [[0, 2000000]]
        for table_name in self.table_list:
            if table_name == 'judgment_minshi_data':
                start_end_df = pd.DataFrame(start_end_mingshi)
            else:
                start_end_df = pd.DataFrame(start_end_xingshi)
            for index, start_and_end in start_end_df.iterrows():
                df_data = self.get_df_data_from_db(table_name, start_and_end[0], start_and_end[1])
                self.insert_data_to_es(df_data, table_name)


    def handle_es(self, df_data, table_name):
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
            ]
            data_body = {key: value for key, value in data_ori.items() if key in use_data}
            data_body["db_name"] = self.db_name
            data_body["table_name"] = table_name
            yield {"_op_type": "create", "_index": self.index_name, "_type": "_doc", "_id": row['id'], "_source": data_body}


if __name__ == '__main__':
    import pandas as pd

    pd.set_option("display.max_columns", None)

    case_es = CaseESTool(**case_es_tools)
    case_es()

    query_dict = {"query": {"match": {"content": "买卖"}}}
    res = case_es.search_data_from_es(query_body=query_dict)

    print(res)
