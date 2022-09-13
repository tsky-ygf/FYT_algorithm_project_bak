#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 09:32
# @Author  : Adolf
# @Site    : 
# @File    : laws_items_es.py
# @Software: PyCharm
import re
import json
from BasicTask.SearchEngine.es_tools import BaseESTool

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
    ],
    "index_name": "flfg",
    "debug": False,
}


# es_hosts = "127.0.0.1:9200"


class LawItemsESTool(BaseESTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.valid_mapping = {"有效": 0, "": 1, "已修改": 2, "尚未生效": 3, "已废止": 4}
        self.legal_mapping = {"法律": 0, "行政法规": 1, "监察法规": 2, "司法解释": 3, "宪法": 5, "地方性法规": 4}

        with open("/home/fyt/huangyulin/project/fyt/ProfessionalSearch/RelevantLaws/LegalLibrary/law_items_graph/minshi_item.json") as f:
            self.minshi_item = json.load(f)

        with open("/home/fyt/huangyulin/project/fyt/ProfessionalSearch/RelevantLaws/LegalLibrary/law_items_graph/xingshi_item.json") as f:
            self.xingshi_item = json.load(f)

    @staticmethod
    def chuli_title(old_s):  # 保留中文、大小写、数字
        cop = re.compile("[^\u4e00-\u9fa5^0-9]")  # 匹配不是中文、大小写、数字的其他字符
        nwe_s = cop.sub("", old_s)  # 将old_s中匹配到的字符替换成空s字符
        return nwe_s

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
            data_body = {key: value for key, value in data_ori.items() if key in use_data}
            data_body["isValid_weight"] = self.valid_mapping[data_body["isValid"]]
            data_body["legal_type_weight"] = self.legal_mapping[data_body["source"]]
            title = self.chuli_title(data_body["title"])
            data_body["title_weight"] = 0
            if title in self.minshi_item:
                data_body["title_weight"] += self.minshi_item[title]
            if title in self.xingshi_item:
                data_body["title_weight"] += self.xingshi_item[title]
            yield {"_index": self.index_name, "_type": "_doc", "_id": row['id'], "_source": data_body}


if __name__ == '__main__':
    laws_es = LawItemsESTool(**case_es_tools)
    laws_es.es_init()

    for table_name_ in laws_es.table_list:
        df_data_ = laws_es.get_df_data_from_db(table_name_)
        laws_es.insert_data_to_es(df_data_)

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
