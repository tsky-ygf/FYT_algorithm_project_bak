#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 14:22 
@Desc    : None
"""
import streamlit as st
from OnlineServer.LawsuitPrejudgment.testing_page.criminal import testing


def _remove_click_state(_count):
    while True:
        key = 'criminal_ask_btn' + str(_count) + '_clicked'
        if key in st.session_state:
            del st.session_state[key]
            _count += 1
            print("remove key:", key)
        else:
            break
    pass


def show_next_qa(_user_input, _question_answers, _factor_sentence_list, _anyou, _event, _count):
    has_next_question, info = testing.get_criminal_result(_user_input,_question_answers, _factor_sentence_list, _anyou, _event)
    if has_next_question:
        btn_key = 'criminal_ask_btn' + str(_count)
        btn_click_key = btn_key + "_clicked"
        next_question_info = info
        next_question = next_question_info["next_question"]
        answers = next_question_info["answers"]
        single_or_multi = next_question_info["single_or_multi"]

        st.markdown('**{}**'.format(next_question))
        if single_or_multi == "single":
            selected_answers = [
                st.radio("", options=answers, key='criminal_aks_' + str(_count), on_change=_remove_click_state, args=(_count,))]
        else:
            selected_answers = []
            for idx, option in enumerate(answers):
                if st.checkbox(option, key='criminal_ask_' + str(_count) + '_item_' + str(idx)):
                    selected_answers.append(option)
        st.write(selected_answers)
        if st.button("确定", key=btn_key):
            st.session_state[btn_click_key] = True
        if btn_click_key in st.session_state:
            _question_answers[next_question + ":" + ";".join(answers)] = ";".join(selected_answers)
            _factor_sentence_list = next_question_info['factor_sentence_list']
            _anyou = next_question_info['anyou']
            _event = next_question_info['event']
            _count += 1
            show_next_qa(_user_input, _question_answers, _factor_sentence_list, _anyou, _event, _count)
    else:
        show_report(info.get("result"))
    pass


def show_report(result):
    st.subheader("产生评估报告")
    report = result["report"][0]
    for item in report:
        st.markdown("**{}**".format(item["title"]))
        if item["type"] == "TYPE_TEXT":
            st.markdown(item["content"], unsafe_allow_html=True)
        elif item["type"] == "TYPE_LIST_OF_TEXT":
            for every_content in item["content"]:
                st.markdown(every_content, unsafe_allow_html=True)
    pass


def criminal_prejudgment_testing_page():
    st.subheader('描述经过')
    user_input = st.text_area('请描述您的纠纷经过，描述越全面评估越准确', '''''', key="criminal_text_area")

    if st.button("提交评估", key="criminal_submit_to_evaluate"):
        st.session_state["criminal_submit_desp"] = True
    if "criminal_submit_desp" in st.session_state:
        st.subheader("进行提问")
        question_answers = {}
        factor_sentence_list = []
        anyou = None
        event = None
        count = 1
        show_next_qa(user_input, question_answers, factor_sentence_list, anyou, event, count)
