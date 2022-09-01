#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 11:52
# @Author  : Adolf
# @Site    :
# @File    : criminal_show.py
# @Software: PyCharm
import streamlit as st

# from LawsuitPrejudgment.Criminal.criminal_judgment_func import criminal_pre_judgment
from LawsuitPrejudgment.Criminal.criminal_prejudgment import CriminalPrejudgment

st.title("刑事诉讼预判")

text = st.text_area(
    "请输入文本",
    value="湖南省涟源市人民检察院指控，2014年8月至2015年1月，被告人刘某甲先后多次容留刘2某、刘某乙、刘1某、刘某丙、袁某等人在其位于本市"
          "安平镇田心村二组的家中吸食甲基苯丙胺（冰毒）和甲基苯丙胺片剂（麻古）。具体事实如下：1、2014年8月份的一天，被告人刘某甲容留刘某"
          "丙、刘1某等人在其家中卧室吸食甲基苯丙胺和甲基苯丙胺片剂。",
    height=200,
)
# run = st.button("开始评估")

criminal_config = {
    "log_level": "debug",
    "prejudgment_type": "criminal",
    "anyou_identify_model_path": "model/gluon_model/accusation",
    "situation_identify_model_path": "http://172.19.82.199:7777/information_result",
}
criminal_pre_judgment = CriminalPrejudgment(**criminal_config)

# run = st.button("开始评估")

input_dict = {"fact": text}
# 第一次调用
res = criminal_pre_judgment(**input_dict)
while "report_result" not in res:
    q_a = res["question_answers"]
    # st.write(q_a)
    for key, value in q_a.items():
        if value["usr_answer"] != "":
            continue
        st.write(value["question"])
        # st.write("请选择答案")
        option = st.radio("请选择答案", value["answer"].split("|"), key='answer')
        res['question_answers'][key]['usr_answer'] = option
        break
    st.write(res)
    res = criminal_pre_judgment(**res)  # 传入上一次的结果
    # break

for key, value in res["report_result"].items():
    st.write("### {}".format(key))
    st.write(value)
