#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 14:03 
@Desc    : None
"""
import requests
import streamlit as st

URL = "http://127.0.0.1:8105"


def _show_administrative_report(type_id, selected_situation):
    result = requests.post(url=URL + "/get_administrative_result",
                           json={"type_id": type_id, "situation": selected_situation}).json().get("result")
    report = result["report"][0]
    for item in report:
        st.markdown("#### {}".format(item["title"]))
        if item["type"] == "TYPE_TEXT":
            st.markdown(item["content"])
        elif item["type"] == "TYPE_LIST_OF_TEXT":
            for every_content in item["content"]:
                st.markdown(every_content)
        elif item["type"] == "TYPE_LIST_OF_OBJECT":
            for every_content in item["content"]:
                first_flag = True
                for key, value in every_content.items():
                    if first_flag:
                        st.markdown("**{}**".format(value))
                        first_flag = False
                    else:
                        st.markdown(value)


def administrative_prejudgment_testing_page():
    supported_administrative_list = requests.get(url=URL+"/get_administrative_type").json().get("result")
    selected_type_name = st.selectbox("请选择你遇到的纠纷类型", [item["type_name"] for item in supported_administrative_list])
    type_id = next((item["type_id"] for item in supported_administrative_list if item["type_name"] == selected_type_name), None)

    problem_and_situation_list = requests.get(url=URL+"/get_administrative_problem_and_situation_by_type_id", params={"type_id": type_id}).json().get("result")
    selected_problem = st.selectbox("请选择你遇到的问题", [item["problem"] for item in problem_and_situation_list], key="一级")
    selected_situation = st.selectbox("请选择具体的情形", next((item["situations"] for item in problem_and_situation_list if item["problem"] == selected_problem), None), key="情形")

    run = st.button("开始计算", key="run")
    if run:
        _show_administrative_report(type_id, selected_situation)
