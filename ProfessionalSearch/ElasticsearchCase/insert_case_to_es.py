#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 09:53
# @Author  : Adolf
# @Site    :
# @File    : insert_case_to_es.py
# @Software: PyCharm
from BasicTask.SearchEngine.es_tools import BaseESTool

case_es_tools = {
    "host": "172.19.82.227",
    "user": "root",
    "passwords": "Nblh@2022",
    "db_name": "judgments_data",
    "table_list": ["judgment_minshi_data", "judgment_xingshi_data"],
    # "db_name": "falvfagui_data",
    # "table_list": ["flfg_result_dfxfg"],
    "index_name": "case_index",
    "debug": True,
}


# es_hosts = "127.0.0.1:9200"


class CaseESTool(BaseESTool):
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
            yield {"_index": self.index_name, "_type": "_doc", "_id": row['id'], "_source": data_body}


if __name__ == '__main__':
    case_es = CaseESTool(**case_es_tools)
    case_es.es_init()

    for table_name_ in case_es.table_list:
        df_data_ = case_es.get_df_data_from_db(table_name_)
        case_es.insert_data_to_es(df_data_, table_name_)

    query_dict = {"query": {"match": {"content": "买卖"}}}
    res = case_es.search_data_from_es(query_body=query_dict)

    print(res)
