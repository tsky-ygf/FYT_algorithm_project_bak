#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 17:43
# @Author  : Adolf
# @Site    : 
# @File    : lib_show.py
# @Software: PyCharm
import streamlit as st
from RelevantLaws.LegalLibrary.read_legal_from_db import search_data_from_es
from RelevantLaws.LegalLibrary import templates
from pprint import pprint

legal_type = st.sidebar.selectbox("请选择法条种类", ["不指定", "宪法"], key="legal_type")
isVaild = st.sidebar.selectbox("选择法律时效性", ["不指定", "有效", "尚未生效", "已修改", "已废止", "缺失"], key="text")
size = st.sidebar.number_input("搜索结果展示条数", value=10, key="size", min_value=1, max_value=100)

text = st.text_input("请输入法条内容", value="", key="text")

run = st.button("查询", key="run")
# print(legal_type)
# print(isVaild)
query_list = []
if isVaild == "缺失":
    isVaild = ''

if run:
    if text is not None:
        query_list.append({"match": {"resultClause": text}})
    if isVaild is not None and isVaild != "不指定":
        query_list.append({"term": {"isValid.keyword": isVaild}})
    if legal_type is not None and isVaild != "不指定":
        query_list.append({"term": {"source.keyword": legal_type}})

    query_dict = {
        "query": {"bool": {"must": query_list}},
        "size": 10,
    }
    pprint(query_dict)
    res = search_data_from_es(query_dict)
    # st.write(res)
    for index, row in res.iterrows():
        # pprint(row.to_dict())
        res_dict = {'标号': index, '标题': row['title'], '法律类别': row['source'], '时效性': row['isValid'],
                    '法条章节': row['resultChapter'], '法条条目': row['resultSection'], '法条内容': row['resultClause'], }
        st.write(res_dict)
        # st.write(row['resultClause'])res_dict
        # st.write(templates.search_result(index, **res), unsafe_allow_html=True)
        # st.write(templates.tag_boxes(text, 'xxxxx', ''), unsafe_allow_html=True)
        #     st.write(templates.search_result(i=index, url="", title=res["title"], highlights=text,
        #                                      author="", length=None, unsafe_allow_html=True))
