#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : show.py
# @Software: PyCharm
import streamlit as st
import requests


def intelligent_consultation_main():
    consult_type = st.sidebar.selectbox("请选择咨询的类型", ["民、商、刑", "专题"], key="咨询类型")

    question = st.text_input("请输入您的问题", value="公司交不起税怎么办", key="问题")

    if consult_type == "民、商、刑":
        res = requests.post("http://127.0.0.1:8130/get_query_answer",
                            json={"question": question}).json()

    elif consult_type == "专题":
        res = requests.post("http://127.0.0.1:8130/get_query_answer_with_source",
                            json={"question": question}).json()
    else:
        res = {"answer": "暂不支持该咨询类型"}

    st.write(res["answer"])
    if "similarity_question" in res:
        for similarity_question in res["similarity_question"]:
            with st.expander(similarity_question["question"], expanded=False):
                st.markdown(similarity_question["answer"], unsafe_allow_html=True)
        # st.markdown(meta.STORY, unsafe_allow_html=True)

        # for similarity_question in res["similarity_question"]:
        #     st.write(similarity_question)
