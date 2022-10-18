#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 09:53
# @Author  : Adolf
# @Site    :
# @File    : insert_case_to_es.py
# @Software: PyCharm
import jieba
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from jieba import analyse

from BasicTask.SearchEngine.es_tools import BaseESTool
import connectorx as cx
from loguru import logger

from ProfessionalSearch.src.similar_case_retrival.similar_case.rank_util import pseg_txt

case_es_tools = {
    "host": "172.19.82.227",
    "user": "root",
    "passwords": "Nblh@2022",
    "db_name": "judgments_data",
    "table_list": [
        # "judgment_xingzheng_data",
        # "judgment_xingshi_data",
        # "judgment_zhixing_data",
        "judgment_minshi_data",
    ],
    "index_name": "case_index_minshi_v2",
    "debug": False,
    "use_big_data": True,
}


# es_hosts = "127.0.0.1:9200"


class CaseESTool(BaseESTool):
    def get_query(self, table_name, start, end):
        if self.debug:
            query = "SELECT A.id, A.uq_id, A.event_num, A.jfType, A.yg_sc, A.bg_sc, A.jslcm, A.byrw, A.pubDate FROM ( select id,uq_id,event_num,jfType,yg_sc,bg_sc,jslcm,byrw,pubDate from {} limit 20 ) A  order by A.pubDate DESC".format(
                table_name
            )
        else:
            query = "select id,uq_id,event_num,jfType,yg_sc,bg_sc,jslcm,byrw,pubDate,wsTitle from {} where pubDate > '2019-01-01' limit {},{}".format(
                table_name, start, end
            )
        logger.info(query)
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
                "doc": data_body,  # update 操作得用doc, index,create 操作用_source
            }

    def __call__(self):
        # self.es_init()
        start_end_mingshi = [
            [0, 2000000],
            [2000000, 2000000],
            [4000000, 2000000],
            [6000000, 2000000],
            [8000000, 2000000],
            [10000000, 2000000],
            [12000000, 1899519],
        ]  # 13899519   尽量插入用uq_id

        # start_end_xingshi = [[0, 2141203]]  # 2141203
        # start_end_xingzheng = [[0, 354424]]  # 354424
        # start_end_zhixing = [
        #     [0, 2000000],
        #     [2000000, 2000000],
        #     [4000000, 3488256],
        # ]  #  7488256

        for table_name in self.table_list:
            # if table_name == "judgment_xingzheng_data":
            #     start_end_df = pd.DataFrame(start_end_xingzheng)
            # elif table_name == "judgment_zhixing_data":
            #     start_end_df = pd.DataFrame(start_end_zhixing)
            if table_name == "judgment_minshi_data":
                start_end_df = pd.DataFrame(start_end_mingshi)
            # elif table_name == "judgment_xingshi_data":
            #     start_end_df = pd.DataFrame(start_end_xingshi)
            for index, start_and_end in start_end_df.iterrows():
                print(table_name, start_and_end[0], start_and_end[1])
                df_data = self.get_df_data_from_db(
                    table_name, start_and_end[0], start_and_end[1]
                )
                self.update_data_from_es_parall(6, 1000, df_data, table_name)

    # def handle_es_parall(self, df_data, table_name):
    #     for index, row in df_data.iterrows():
    #         data_ori = row.to_dict()
    #         use_data = [
    #             "id",
    #             "uq_id",
    #             # "content",
    #             "event_num",
    #             "faYuan_name",
    #             "jfType",
    #             # "event_type",
    #             # "province",
    #             "yg_sc",
    #             "bg_sc",
    #             "jslcm",
    #             "byrw",
    #             "pubDate",
    #         ]
    #         data_body = {
    #             key: value for key, value in data_ori.items() if key in use_data
    #         }
    #         # data_body["db_name"] = self.db_name
    #         # data_body["table_name"] = table_name
    #         yield {
    #             "_index": self.index_name,
    #             "_id": row["uq_id"],
    #             "_type": "_doc",
    #             "_source": data_body,
    #         }

    def handle_es_parall(self, df_data, table_name):
        count = 0
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                "id",
                "uq_id",
                # "content",
                "event_num",
                "faYuan_name",
                "jfType",
                # "event_type",
                # "province",
                # "yg_sc",
                # "bg_sc",
                # "jslcm",
                # "byrw",
                "pubDate",
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            yg_sc_sentences, bg_sc_sentences = '', ''
            if row["yg_sc"]:
                yg_sc_sentences = bytes.decode(row["yg_sc"]).strip()
                yg_sc_sentences = yg_sc_sentences[0:400]
                yg_sc_sentences = pseg_txt(yg_sc_sentences)
            else:
                yg_sc_sentences = ""

            if row["bg_sc"]:
                bg_sc_sentences = bytes.decode(row["bg_sc"]).strip()
                bg_sc_sentences = bg_sc_sentences[0:400]
                bg_sc_sentences = pseg_txt(bg_sc_sentences)
            else:
                bg_sc_sentences = ""

            if bg_sc_sentences or yg_sc_sentences:
                sucheng_sentences = bg_sc_sentences + " " + yg_sc_sentences
                sucheng_sentences = sucheng_sentences[0:400]
                sucheng_sentences = pseg_txt(sucheng_sentences)
                data_body["sucheng_sentences"] = sucheng_sentences
            else:
                data_body["sucheng_sentences"] = bg_sc_sentences + " " + yg_sc_sentences

            if row["jslcm"]:
                chaming = bytes.decode(row["jslcm"]).strip()
                chaming = chaming[0:400]
                chaming = pseg_txt(chaming)
                data_body["chaming"] = chaming
            else:
                data_body["chaming"] = row["jslcm"]

            if row["byrw"]:
                benyuan_renwei = bytes.decode(row["byrw"]).strip()  # 本院认为
                benyuan_renwei = benyuan_renwei[0:400]
                benyuan_renwei = pseg_txt(benyuan_renwei)
                data_body["benyuan_renwei"] = benyuan_renwei
            else:
                data_body["benyuan_renwei"] = row["byrw"]

            sucheng_sentences = data_body["sucheng_sentences"]
            if not sucheng_sentences:
                sucheng_sentences = ""
            chaming = data_body["chaming"]
            if not chaming:
                chaming = ""
            benyuan_renwei = data_body["benyuan_renwei"]
            if not benyuan_renwei:
                benyuan_renwei = ""
            query_all = sucheng_sentences + " " + chaming + " " + benyuan_renwei
            tags1 = analyse.extract_tags(query_all, topK=15)
            tags2 = analyse.textrank(
                query_all, topK=15, withWeight=False, allowPOS=("ns", "n", "vn", "v")
            )
            tags = list(set(tags1).intersection(set(tags2)))
            tags = [tag for tag in tags if tag not in ["原告", "被告", "双方", "诉至", "判决"]]
            tags = " ".join(tags)

            if count < 30:
                print("doc_id:", row["uq_id"], ";problem_type:", row["jfType"])
                print("suqing_sentences:", sucheng_sentences)
                print("chaming:", chaming)
                print("benyuan_renwei:", benyuan_renwei)
                print("tags:", tags)
                print("==============================")
            if count % 10000 == 0:
                print(count)
            count = count + 1
            print("count:" + str(count))

            data_body["tags"] = tags
            yield {
                "_index": self.index_name,
                "_id": row["uq_id"],
                "_type": "_doc",
                "_source": data_body,
            }

    def update_es_parall(self, df_data, table_name):
        count = 0
        for index, row in df_data.iterrows():
            data_ori = row.to_dict()
            use_data = [
                "id",
                "uq_id",
                # "content",
                "event_num",
                "faYuan_name",
                "jfType",
                # "event_type",
                # "province",
                # "yg_sc",
                # "bg_sc",
                # "jslcm",
                # "byrw",
                "pubDate",
                "wsTitle"
            ]
            data_body = {
                key: value for key, value in data_ori.items() if key in use_data
            }
            yg_sc_sentences, bg_sc_sentences = '', ''
            if row["yg_sc"]:
                yg_sc_sentences = bytes.decode(row["yg_sc"]).strip()
                yg_sc_sentences = yg_sc_sentences[0:400]
                yg_sc_sentences = pseg_txt(yg_sc_sentences)
            else:
                yg_sc_sentences = ""

            if row["bg_sc"]:
                bg_sc_sentences = bytes.decode(row["bg_sc"]).strip()
                bg_sc_sentences = bg_sc_sentences[0:400]
                bg_sc_sentences = pseg_txt(bg_sc_sentences)
            else:
                bg_sc_sentences = ""

            if bg_sc_sentences or yg_sc_sentences:
                sucheng_sentences = bg_sc_sentences + " " + yg_sc_sentences
                sucheng_sentences = sucheng_sentences[0:400]
                sucheng_sentences = pseg_txt(sucheng_sentences)
                data_body["sucheng_sentences"] = sucheng_sentences
            else:
                data_body["sucheng_sentences"] = str(bg_sc_sentences + " " + yg_sc_sentences, encodings='utf-8')

            if row["jslcm"]:
                chaming = bytes.decode(row["jslcm"]).strip()
                chaming = chaming[0:400]
                chaming = pseg_txt(chaming)
                data_body["chaming"] = chaming
            else:
                data_body["chaming"] = str(row["jslcm"], encodings='utf-8')

            if row["byrw"]:
                benyuan_renwei = bytes.decode(row["byrw"]).strip()  # 本院认为
                benyuan_renwei = benyuan_renwei[0:400]
                benyuan_renwei = pseg_txt(benyuan_renwei)
                data_body["benyuan_renwei"] = benyuan_renwei
            else:
                data_body["benyuan_renwei"] = str(row["byrw"], encodings='utf-8')

            sucheng_sentences = data_body["sucheng_sentences"]
            if not sucheng_sentences:
                sucheng_sentences = ""
            chaming = data_body["chaming"]
            if not chaming:
                chaming = ""
            benyuan_renwei = data_body["benyuan_renwei"]
            if not benyuan_renwei:
                benyuan_renwei = ""
            query_all = sucheng_sentences + " " + chaming + " " + benyuan_renwei
            tags1 = analyse.extract_tags(query_all, topK=15)
            tags2 = analyse.textrank(
                query_all, topK=15, withWeight=False, allowPOS=("ns", "n", "vn", "v")
            )
            tags = list(set(tags1).intersection(set(tags2)))
            tags = [tag for tag in tags if tag not in ["原告", "被告", "双方", "诉至", "判决"]]
            tags = " ".join(tags)

            if count < 30:
                print("doc_id:", row["uq_id"], ";problem_type:", row["jfType"])
                print("suqing_sentences:", sucheng_sentences)
                print("chaming:", chaming)
                print("benyuan_renwei:", benyuan_renwei)
                print("tags:", tags)
                print("==============================")
            if count % 10000 == 0:
                print(count)
            count = count + 1

            data_body["tags"] = str(tags, encodings='utf-8')
            yield {
                "_op_type": "update",
                "_index": self.index_name,
                "_type": "_doc",
                "_id": row["uq_id"],
                "doc": data_body,
            }


def insert_case_to_es():
    import pandas as pd

    pd.set_option("display.max_columns", None)

    case_es = CaseESTool(**case_es_tools)
    case_es()

    query_dict = {"query": {"match": {"content": "买卖"}}}
    res = case_es.search_data_from_es(query_body=query_dict)

    print(res)


if __name__ == '__main__':
    insert_case_to_es()
