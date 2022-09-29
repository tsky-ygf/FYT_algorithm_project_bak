#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/26 14:17
# @Author  : Adolf
# @Site    :
# @File    : read_legal_from_db.py
# @Software: PyCharm
import addressparser
import pymysql
import pandas as pd

# from pprint import pprint
# from sqlalchemy.engine import URL
# from sqlalchemy import create_engine
from elasticsearch import Elasticsearch, helpers
import json
import re

pd.set_option("display.max_columns", None)

__all__ = [
    "connect_mysql",
    "get_df_from_sql",
    "es_init",
    "insert_data_to_es",
    "search_data_from_es",
]

host = "172.19.82.227"
user = "root"
passwords = "Nblh@2022"
db = "big_data_ceshi227"


def connect_mysql(_host, _user, _passwd, _db):
    # 连接mysql
    connect_big_data = pymysql.connect(
        host=_host, user=_user, password=_passwd, db=_db, charset="utf8"
    )

    # sever_string = f"SERVER={host};DATABASE={db};UID={user};PWD={passwords}"
    # connection_string = "DRIVER={ODBC Driver 17 for SQL Server};" + sever_string
    # connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
    # engine = create_engine(connection_url)
    # return engine
    return connect_big_data


index_name = "flfg"

# legal_type = st.sidebar.selectbox("请选择法条种类", ["不指定", "宪法"], key="legal_type")
# isValid = st.sidebar.selectbox("选择法律时效性", ["不指定", "有效", "已修改", "尚未生效", "已废止"], key="text")
valid_mapping = {"有效": 0, "": 1, "已修改": 2, "尚未生效": 3, "已废止": 4}
# legal_mapping = {'宪法': 0, '法律': 1, '行政法规': 2, '监察法规': 3, '司法解释': 4, '地方法律法规': 5}
legal_mapping = {"法律": 0, "行政法规": 1, "监察法规": 2, "司法解释": 3, "宪法": 5, "地方性法规": 4}

db_name = "falvfagui_data"
table_name_list = [
    "flfg_result_dfxfg",
    "flfg_result_falv",
    "flfg_result_sfjs",
    "flfg_result_xf",
    "flfg_result_xzfg",
]

# with open("RelevantLaws/LegalLibrary/law_items_graph/minshi_item.json") as f:
#     minshi_item = json.load(f)
#
# with open("RelevantLaws/LegalLibrary/law_items_graph/xingshi_item.json") as f:
#     xingshi_item = json.load(f)


# print(xingshi_item)
# exit()

# sqlcmd = """select * from test_falvfagui_data.test_flfg_result_xf"""
# sqlcmd = """select * from test_falvfagui_data.test_flfg_result_xzfg"""
# sqlcmd = """select * from test_falvfagui_data.test_flfg_result_falv"""


def get_df_from_sql(_table_name, _db_name):
    connect_big_data = connect_mysql(
        _host=host, _user=user, _passwd=passwords, _db=db_name
    )
    cursor = connect_big_data.cursor()
    # 获取数据
    print("sql查询开始")
    sqlcmd = """select uq_id, content, event_num, faYuan_name, jfType, event_type from {}.{} limit 0, 2000000""".format(
        _db_name, _table_name
    )
    # df = pd.read_sql(sqlcmd, connect_big_data)
    cursor.execute(sqlcmd)
    # df = cursor.fetchall()
    df = pd.DataFrame(cursor.fetchall())
    df.columns = [one[0] for one in cursor.description]
    cursor.close()
    print("sql查询完成")
    return df


def update_flfg_dfxfg(_table_name, _db_name):
    connect_big_data = connect_mysql(
        _host=host, _user=user, _passwd=passwords, _db=db_name
    )
    cursor = connect_big_data.cursor()
    # 获取数据
    print("sql查询开始")
    sqlcmd = """select id, locality from {}.{} where prov='' limit 0, 2000000""".format(
        _db_name, _table_name
    )
    # df = pd.read_sql(sqlcmd, connect_big_data)
    cursor.execute(sqlcmd)
    # df = cursor.fetchall()
    df_old_pro = pd.DataFrame(cursor.fetchall())
    for index_old_pro, row_old_pro in df_old_pro.iterrows():
        # new_prov = addressparser.transform([row_old_pro[1]])
        new_prov = get_mapping(row_old_pro[1])
        sqlcmd = "update flfg_result_dfxfg set prov = '{}' where id = '{}' ".format(
            str(new_prov), row_old_pro[0]
        )
        cursor.execute(sqlcmd)
    connect_big_data.commit()
    # df_old_pro.columns = [one[0] for one in cursor.description]
    cursor.close()
    print("sql查询完成")
    connect_big_data.close()
    return df_old_pro


def get_mapping(row_old_pro):
    prov_spec_json = {
        "德宏傣族景颇族自治州": "云南省",
        "德宏傣族景颇族自治州": "云南省",
        "大通回族土族自治县": "青海省",
        "黔南布依族苗族自治州": "贵州省",
        "民和回族土族自治县": "青海省",
        "海西蒙古族藏族自治州": "青海省",
        "喀喇沁左翼蒙古族自治县": "辽宁省",
        "镇宁布依族苗族自治县": "贵州省",
        "邓小平故里历史文化遗存保护条例": "四川省",
        "塞罕坝森林草原防火条例": "河北省",
        "黔东南苗族侗族自治州": "贵州省",
        "关岭布依族苗族自治县": "贵州省",
        "白沙黎族自治县": "海南省",
        "道真仡佬族苗族自治县": "贵州省",
        "博尔塔拉蒙古自治州": "新疆维吾尔自治区",
        "克孜勒苏柯尔克孜自治州": "克孜勒苏柯尔克孜自治州",
        "泰山风景名胜区": "山东省",
        "湘西土家族苗族自治州": "湖南省",
        "印江土家族苗族自治县": "贵州省",
        "西双版纳傣族自治州": "云南省",
        "郑洛新国家自主创新示范区": "河南省",
        "务川仡佬族苗族自治县": "贵州省",
        "赣南客家围屋保护条例": "江西省",
        "恒山风景名胜区": "山西省",
        "威宁彝族回族苗族自治县": "贵州省",
        "酉阳土家族苗族自治县": "重庆市",
        "巴音郭楞蒙古自治州": "新疆维吾尔自治区",
        "巴音布鲁克草原生态保护条例": "新疆维吾尔自治区",
        "保亭黎族苗族自治县": "海南省",
        "彭水苗族土家族自治县": "重庆市",
        "伊犁河谷生态环境保护条例": "新疆维吾尔自治区",
        "江布拉克景区": "新疆维吾尔自治区",
        "九华山风景名胜区": "安徽省",
        "云冈石窟保护条例": "山西省",
        "前郭尔罗斯蒙古族自治县": "吉林省",
        "乐东黎族自治县": "海南省",
        "清东陵保护管理办法": "河北省",
        "神农架国家公园保护条例": "湖北省",
        "库鲁斯台草原生态保护条例": "新疆维吾尔自治区",
        "陵水黎族自治县": "海南省",
        "紫云苗族布依族自治县": "贵州省",
        "昌江黎族自治县": "海南省",
        "关于落实电信用户真实身份信息登记制度的决定": "广东省",
        "东湖国家自主创新示范区": "湖北省",
        "秀山土家族苗族自治县": "重庆市",
    }
    # prov_spec = json.loads(prov_spec_json)
    return prov_spec_json[row_old_pro]


def es_init(_index_name="flfg", _es_hosts="127.0.0.1:9200"):
    es = Elasticsearch(hosts=_es_hosts)
    # 重新创建索引
    es.indices.delete(index=_index_name, ignore=[400, 404])
    es.indices.create(index=_index_name, ignore=400)


def chuli_title(old_s):  # 保留中文、大小写、数字
    cop = re.compile("[^\u4e00-\u9fa5^0-9]")  # 匹配不是中文、大小写、数字的其他字符
    nwe_s = cop.sub("", old_s)  # 将old_s中匹配到的字符替换成空s字符
    return nwe_s


# 构造es插入迭代器
# def handle_es(df_data, *args, **kwargs):
#     # 构造迭代器
#     for index, row in df_data.iterrows():
#         data_ori = row.to_dict()
#         use_data = [
#             "isValid",
#             "resultChapter",
#             "resultClause",
#             "resultSection",
#             "title",
#             "md5Clause",
#             "source",
#             "locality",
#         ]
#         data_body = {key: value for key, value in data_ori.items() if key in use_data}
#         data_body["isValid_weight"] = valid_mapping[data_body["isValid"]]
#         data_body["legal_type_weight"] = legal_mapping[data_body["source"]]
#         title = chuli_title(data_body["title"])
#         data_body["title_weight"] = 0
#         if title in minshi_item:
#             data_body["title_weight"] += minshi_item[title]
#         if title in xingshi_item:
#             data_body["title_weight"] += xingshi_item[title]
#         yield {"_index": index_name, "_type": "_doc", "_source": data_body}


#
# inset_data
# def insert_data_to_es(
#     _es_hosts="127.0.0.1:9200",
#     _db_name=db_name,
#     _table_name_list=table_name_list,
#     _handle_es=handle_es,
# ):
#     es = Elasticsearch(hosts=_es_hosts)
#
#     for table_name in _table_name_list:
#         df_data = get_df_from_sql(_db_name=_db_name, _table_name=table_name)
#         # 插入数据
#         helpers.bulk(es, _handle_es(df_data, _db_name=db_name, _table_name=table_name))


#


def search_data_from_es(query_body, _index_name="flfg", _es_hosts="127.0.0.1:9200"):
    # 查询数据
    es = Elasticsearch(hosts=_es_hosts)
    res = es.search(index=_index_name, body=query_body)
    # print("Got %d Hits:" % res['hits']['total']['value'])
    res_list = [hit["_source"] for hit in res["hits"]["hits"]]
    df = pd.DataFrame(res_list)
    df.fillna("", inplace=True)
    # pprint(res_list[0])
    # sort_list = ["有效", "已修改", "尚未生效", "已废止"]
    # df.index = df['isValid']
    # sort_df_grade = df.loc[sort_list]
    return df, res["hits"]["total"]["value"]
    # for index, hit in enumerate(res['hits']['hits']):
    #     print(index)
    #     print(hit["_source"])

    # return res


if __name__ == "__main__":
    # es_init()
    # es = Elasticsearch(hosts="127.0.0.1:9200")
    # print(es.cat.indices())
    # insert_data_to_es()
    # query_dict = {
    #     "query": {
    #         "bool": {
    #             "must": [
    #                 {
    #                     "bool": {
    #                         "should": [
    #                             {"match_phrase": {"resultClause": "涉外"}},
    #                             {"match_phrase": {"title": "涉外"}},
    #                         ]
    #                     }
    #                 },
    #                 {
    #                     "bool": {
    #                         "should": [
    #                             {"match_phrase": {"resultClause": "离婚"}},
    #                             {"match_phrase": {"title": "离婚"}},
    #                         ]
    #                     }
    #                 },
    #                 {"terms": {"isValid.keyword": ["有效"]}},
    #                 {"terms": {"source.keyword": ["法律"]}},
    #             ]
    #         }
    #     },
    #     "size": 10,
    #     "sort": [
    #         {"title_weight": {"order": "desc"}},
    #         {"isValid_weight": {"order": "asc"}},
    #         {"legal_type_weight": {"order": "asc"}},
    #     ],
    # }
    #
    # # # print(get_df_from_sql(table_name_list[0]))
    #
    # # pd.set_option('display.width', 1000)
    # # res_df = search_data_from_es({"query": {"match_all": {}}, "size": 10})
    query_dict = {
        "from": 1,
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    {"match_phrase": {"source": {"query": "地方性法规", "boost": 5}}},
                    {"match_phrase": {"isValid": {"query": "全部", "boost": 5}}},
                    {"match_phrase": {"prov": {"query": "福建省", "boost": 5}}}
                ]
            }
        },
        # "sort": [
        #     {"title_weight": {"order": "desc"}},
        #     {"isValid_weight": {"order": "asc"}},
        #     {"legal_type_weight": {"order": "asc"}},
        # ],
    }
    res_df, total = search_data_from_es(query_dict)
    for index, row in res_df.iterrows():
        print(row.to_dict())
    # #     print(row['resultClause'])
    # #     print('-' * 100)
    # _db_name, _table_name = "falvfagui_data", "flfg_result_dfxfg"
    # update_flfg_dfxfg(_table_name, _db_name)
