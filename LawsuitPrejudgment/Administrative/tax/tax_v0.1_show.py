#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 10:34
# @Author  : Adolf
# @Site    : 
# @File    : tax_v0.1_show.py
# @Software: PyCharm
import json
# import pandas as pd
import streamlit as st
from pprint import pprint
from annotated_text import annotated_text

st.markdown("# 税务处罚预判")

# st.write("请选择你遇到的问题：")
# tax_config = pd.read_csv("LawsuitPrejudgment/Administrative/tax/tax_config.csv")


with open('LawsuitPrejudgment/Administrative/tax/tax_config.json', 'r') as f1:
    info_data = json.load(f1)

with open('LawsuitPrejudgment/Administrative/tax/tax_type.json', 'r') as f2:
    type_data = json.load(f2)

# pprint(type_data)
one = st.selectbox("请选择你遇到的问题", type_data.keys(), key="一级")

situation_list = []
for key, value in type_data[one].items():
    situation_list += value

# if run:
# situation_list = [value for key, value in type_data[one].items()]
situation = st.selectbox("请选择具体的情形", situation_list, key="情形")
run = st.button("开始计算", key="run")

if run:
    # pprint(info_data[situation])
    st.markdown("### 一、具体情形")
    st.write('{}({})'.format(situation, info_data[situation]['法条类别']))

    st.markdown("### 二、涉嫌违法行为")
    for yiju in info_data[situation]['处罚依据']:
        st.markdown("- {}".format(yiju))
    # st.write(info_data[situation]["处罚依据"])

    st.markdown("### 三、法条依据")
    # st.write(info_data[situation]['法条依据'])
    for one_law in info_data[situation]['法条依据']:
        st.write("###### {}".format(one_law))
        one_law_list = info_data[situation]['法条依据'][one_law].replace('【', '|').replace('】', '|').split('|')
        # res_txt = ""
        try:
            annotated_text(one_law_list[0], (one_law_list[1], "处罚依据", "#8ef"), one_law_list[2])
        except Exception as e:
            print(e)
            st.markdown(info_data[situation]['法条依据'][one_law])
        # annotated_text(["This ", ("is", "处罚依据", "#8ef"), " some "])
        # st.markdown(original_title, unsafe_allow_html=True)
        st.write("")

    st.markdown("### 四、处罚种类")
    # st.write(info_data[situation]['处罚种类'])
    for chufa in info_data[situation]['处罚种类']:
        st.markdown("- {}".format(chufa))

    st.markdown("### 五、处罚幅度")

    for fudu in info_data[situation]['处罚幅度']:
        st.markdown("- {}".format(fudu.replace("\n", "")))

    st.markdown("### 六、涉刑风险")

    for index, fenxian in enumerate(info_data[situation]['涉刑风险']):
        st.write("###### {}、{}".format(index + 1, fenxian))
        if fenxian == '暂无':
            continue
        # st.markdown("- {}".format(fenxian.replace("\n", "")))
        st.write('刑法内容:')
        # st.write(info_data[situation]['涉刑风险'][fenxian])
        st.write(':'.join(info_data[situation]['涉刑风险'][fenxian]))
        # st.write(info_data[situation]['法条依据'][one_law])
        st.write("")

    st.markdown("### 七、相似类案")
    for index, an_case in enumerate(info_data[situation]['相关案例']):
        # st.write()
        st.write(an_case)
        st.write("#" * 50)
