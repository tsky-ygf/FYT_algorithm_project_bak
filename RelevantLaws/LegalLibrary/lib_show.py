#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 17:43
# @Author  : Adolf
# @Site    : 
# @File    : lib_show.py
# @Software: PyCharm
import streamlit as st
from RelevantLaws.LegalLibrary.read_legal_from_db import search_data_from_es
from pprint import pprint

# legal_type = st.sidebar.selectbox("请选择法条种类", ["不指定", "宪法"], key="legal_type")
st.sidebar.write('选择法条种类')
xf = st.sidebar.checkbox('宪法', value=True, key='宪法')
fv = st.sidebar.checkbox('法律', value=True, key='法律')
xzfg = st.sidebar.checkbox('行政法规', value=True, key='行政法规')
jcfg = st.sidebar.checkbox('监察法规', value=False, key='监察法规')
sfjs = st.sidebar.checkbox('司法解释', value=True, key='司法解释')
dfxfg = st.sidebar.checkbox('地方性法规', value=True, key='地方性法规')
st.sidebar.write('选择时效性')
yx = st.sidebar.checkbox('有效', value=True, key='有效')
yxg = st.sidebar.checkbox('已修改', value=True, key='已修改')
swsx = st.sidebar.checkbox('尚未生效', value=True, key='尚未生效')
yfz = st.sidebar.checkbox('已废止', value=True, key='已废止')
# isValid = st.sidebar.selectbox("选择法律时效性", ["不指定", "有效", "已修改", "尚未生效", "已废止"], key="text")
size = st.sidebar.number_input("搜索结果展示条数", value=10, key="size", min_value=1, max_value=100)

text = st.text_input("请输入法条内容", value="", key="text")

run = st.button("查询", key="run")
# print(legal_type)
# print(isValid)
query_list = []

if run:
    if text is not None:
        query_list.append({"match": {"resultClause": text.replace(" ", "")}})
    if isValid is not None and isValid != "不指定":
        query_list.append({"term": {"isValid.keyword": isValid}})
    if legal_type is not None and legal_type != "不指定":
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
        st.write("-" * 20 + "我是分割线" + "-" * 20)
        # st.write(row['resultClause'])res_dict
        # st.write(templates.search_result(index, **res), unsafe_allow_html=True)
        #     st.write(templates.search_result(i=index, url="", title=res["title"], highlights=text,
        #                                      author="", length=None, unsafe_allow_html=True))
