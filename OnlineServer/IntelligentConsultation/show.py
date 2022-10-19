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
    tab1, tab2 = st.tabs(["民、商、刑", "专题"])

    with tab1:
        try:
            question = st.text_input("请输入您的问题", value="公司交不起税怎么办", key="问题1")
            res = requests.post("http://127.0.0.1:8134/get_query_answer_with_source",
                                json={"question": question, "query_type": "旧版"}).json()

        # st.write(res)
            st.write(res["answer"])
        except:
            st.write("旧版接口存在问题！！！")

    with tab2:
        question = st.text_input("请输入您的问题", value="公司交不起税怎么办", key="问题2")
        res = requests.post("http://127.0.0.1:8134/get_query_answer_with_source",
                            json={"question": question, "query_type": "专题"}).json()

        st.write(res["answer"])

        for similarity_question in res["similarity_question"]:
            with st.expander(similarity_question["question"], expanded=False):
                st.markdown(similarity_question["answer"], unsafe_allow_html=True)
