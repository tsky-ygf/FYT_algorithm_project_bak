from pprint import pprint

import jieba
import requests
import streamlit as st
from Utils.logger import Logger

from ProfessionalSearch.RelevantLaws.LegalLibrary.read_legal_from_db import (
    search_data_from_es,
)

logger = Logger(name="law-lib", level="debug").logger
url_law_conditions = "http://172.19.82.199:8161/get_filter_conditions_of_law"
resp_conditions_json = requests.get(url_law_conditions).json()

legal_type = st.sidebar.selectbox(
    "请选择法条种类", resp_conditions_json["result"].get("types_of_law").get("value"), key="legal_type"
)
# st.sidebar.write("选择法条种类")
# qb_ft = st.sidebar.checkbox("全部", value=True, key="全部")
# xf = st.sidebar.checkbox("宪法", value=True, key="宪法")
# fv = st.sidebar.checkbox("法律", value=True, key="法律")
# xzfg = st.sidebar.checkbox("行政法规", value=True, key="行政法规")
# jcfg = st.sidebar.checkbox("监察法规", value=False, key="监察法规")
# sfjs = st.sidebar.checkbox("司法解释", value=True, key="司法解释")
# dfxfg = st.sidebar.checkbox("地方性法规", value=True, key="地方性法规")
# st.sidebar.write("选择时效性")
# qb_sx = st.sidebar.checkbox("全部", value=True, key="全部")
# yx = st.sidebar.checkbox("有效", value=True, key="有效")
# yxg = st.sidebar.checkbox("已修改", value=True, key="已修改")
# swsx = st.sidebar.checkbox("尚未生效", value=True, key="尚未生效")
# yfz = st.sidebar.checkbox("已废止", value=True, key="已废止")
isValid_type = st.sidebar.selectbox(
    "请选择法律时效性", resp_conditions_json["result"].get("timeliness").get("value"), key="text"
)
prov_type = st.sidebar.selectbox(
    "请选择使用范围",
    resp_conditions_json["result"].get("scope_of_use").get("value"),
    key="text",
)
size = st.sidebar.number_input(
    "搜索结果展示条数", value=10, key="size", min_value=1, max_value=100
)

text = st.text_input("请输入法条内容", value="", key="text")

run = st.button("查询", key="run")
# print(legal_type)
# print(isValid)
sxx_list = []
# if qb_sx:
#     sxx_list.append("全部")
# if yx:
#     sxx_list.append("有效")
# if yxg:
#     sxx_list.append("已修改")
# if swsx:
#     sxx_list.append("尚未生效")
# if yfz:
#     sxx_list.append("已废止")

legal_list = []
# if qb_ft:
#     legal_list.append("全部")
# if xf:
#     legal_list.append("宪法")
# if fv:
#     legal_list.append("法律")
# if xzfg:
#     legal_list.append("行政法规")
# if jcfg:
#     legal_list.append("监察法规")
# if sfjs:
#     legal_list.append("司法解释")
# if dfxfg:
#     legal_list.append("地方性法规")

query_list = []
if run:
    url = "http://172.19.82.199:8161/search_laws"
    body = {
        "query": text,
        "filter_conditions": {
            "types_of_law": [
                legal_type
            ],
            "timeliness": [
                isValid_type
            ],
            "scope_of_use": [
                prov_type
            ]
        },
        "page_number": 1,
        "page_size": size
    }
    resp_json = requests.post(url, json=body).json()

    for index, row in enumerate(resp_json.get("result")):
        logger.info(row)
        # break
        # if row["timeliness"] == "有效":
        #     row["timeliness"] = "现行有效"
        # if row["using_range"] == "":
        #     row["using_range"] = "全国"
        res_dict = {
            "标号": index,
            # "md5": row["md5Clause"],
            "标题": row["law_name"],
            "法律类别": row["law_type"],
            "时效性": row["timeliness"],
            "使用范围": row["using_range"],
            "法条章节": row["law_chapter"],
            "法条条目": row["law_item"],
            "法条内容": row["law_content"],
        }
        st.write(res_dict)
        st.write("-" * 20 + "我是分割线" + "-" * 20)

    # text = " ".join(jieba.cut(text))
    # logger.info(text)
    # text_list = text.split(" ")
    #
    # logger.info("查询的".format(text_list))
    #
    # if len(text_list) > 0:
    #     for one_text in text_list:
    #         # query_list.append({'multi_match': {"query": one_text, "fields": ["title", "resultClause"],
    #         #                                    "minimum_should_match": "100%"}})、
    #         query_list.append(
    #             {
    #                 "bool": {
    #                     "should": [
    #                         {"match_phrase": {"resultClause": one_text}},
    #                         {"match_phrase": {"title": one_text}},
    #                     ]
    #                 }
    #             }
    #         )
    # if isValid_type != '全部':
    #     query_list.append(
    #         {"match_phrase": {"isValid": {"query": isValid_type, "boost": 5}}}
    #     )
    # if legal_type != '全部':
    #     query_list.append({"match_phrase": {"source": {"query": legal_type, "boost": 5}}})
    #
    # if prov_type != '全国':
    #     query_list.append({"match_phrase": {"prov": {"query": prov_type, "boost": 5}}})
    #
    # query_dict = {
    #     "query": {"bool": {"must": query_list, }},
    #     # "sort": [
    #     #     {"title_weight": {"order": "desc"}},
    #     #     {"isValid_weight": {"order": "asc"}},
    #     #     {"legal_type_weight": {"order": "asc"}},
    #     # ],
    #     "size": size,
    # }
    #
    # print("查询语句:")
    # pprint(query_dict)
    # print("-" * 50)
    # res = search_data_from_es(query_dict)
    # st.write(res)
    # for index, row in res.iterrows():
    #     # pprint(row.to_dict())
    #     # break
    #     if row["isValid"] == "有效":
    #         row["isValid"] = "现行有效"
    #     if row["prov"] == "":
    #         row["prov"] = "全国"
    #     res_dict = {
    #         "标号": index,
    #         "md5": row["md5Clause"],
    #         "标题": row["title"],
    #         "法律类别": row["source"],
    #         "时效性": row["isValid"],
    #         "使用范围": row["prov"],
    #         "法条章节": row["resultChapter"],
    #         "法条条目": row["resultSection"],
    #         "法条内容": row["resultClause"],
    #     }
    #     st.write(res_dict)
    #     st.write("-" * 20 + "我是分割线" + "-" * 20)
