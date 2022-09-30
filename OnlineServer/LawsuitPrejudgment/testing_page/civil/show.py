#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : show.py
# @Software: PyCharm
import streamlit as st
import pandas as pd
from OnlineServer.LawsuitPrejudgment.testing_page.civil import testing


def _remove_click_state(_count):
    while True:
        key = 'ask_btn' + str(_count) + '_clicked'
        if key in st.session_state:
            del st.session_state[key]
            _count += 1
            print("remove key:", key)
        else:
            break
    pass


def show_debug_info(debug_info):
    st.markdown('**中间信息**')
    st.write(debug_info)
    for k, v in debug_info.items():
        st.markdown(k)
        li = str(v).split('\n')
        for item in li:
            st.markdown(item)


def show_next_qa(_selected_anyou, _selected_suqiu_list, _user_input, _question_answers, _factor_sentence_list, _count, _repeated_question_management):
    has_next_question, info = testing.get_next_question(_selected_anyou, _selected_suqiu_list, _user_input,
                                                        _question_answers, _factor_sentence_list, _repeated_question_management)
    show_debug_info(info.get("debug_info"))
    if has_next_question:
        btn_key = 'ask_btn' + str(_count)
        btn_click_key = btn_key + "_clicked"
        next_question_info = info
        next_question = next_question_info["next_question"]
        answers = next_question_info["answers"]
        single_or_multi = next_question_info["single_or_multi"]

        st.markdown('**{}**'.format(next_question))
        if single_or_multi == "single":
            selected_answers = [
                st.radio("", options=answers, key='aks_' + str(_count), on_change=_remove_click_state, args=(_count,))]
        else:
            selected_answers = []
            for idx, option in enumerate(answers):
                if st.checkbox(option, key='ask_' + str(_count) + '_item_' + str(idx)):
                    selected_answers.append(option)
        st.write(selected_answers)
        if st.button("确定", key=btn_key):
            st.session_state[btn_click_key] = True
        if btn_click_key in st.session_state:
            _question_answers[next_question + ":" + ";".join(answers)] = ";".join(selected_answers)
            _factor_sentence_list = next_question_info['factor_sentence_list']
            _repeated_question_management = next_question_info['repeated_question_management']
            _count += 1
            show_next_qa(_selected_anyou, _selected_suqiu_list, _user_input, _question_answers, _factor_sentence_list,
                         _count, _repeated_question_management)
    else:
        show_report(info.get("result"))
    pass


def show_report(results):
    st.subheader("产生评估报告")
    for suqiu, result in results.items():
        st.markdown("**诉求**")
        st.write(suqiu)
        st.markdown("**评估理由**")
        st.markdown(result['reason_of_evaluation'], unsafe_allow_html=True)
        st.markdown("**证据材料**")
        st.markdown(result['evidence_module'], unsafe_allow_html=True)
        st.markdown("**法律建议**")
        st.markdown(result['legal_advice'], unsafe_allow_html=True)
    pass


def show_extracted_features(_selected_anyou, _selected_suqiu_list, _user_input):
    st.subheader("从用户描述提取到的特征")
    df = pd.DataFrame(data=testing.get_extracted_features(_selected_anyou, _selected_suqiu_list, _user_input))
    st.dataframe(df)


def civil_prejudgment_testing_page():
    selected_anyou = st.sidebar.selectbox(
        label="案由",
        options=testing.get_anyou_list()
    )


    st.subheader('描述经过')
    user_input = st.text_area('请描述您的纠纷经过，描述越全面评估越准确', '''''')
    st.subheader("选择诉求")
    selected_suqiu_list = st.multiselect(
        '',
        testing.get_suqiu_list(selected_anyou),
        None)

    if st.button("提交评估"):
        st.session_state["submit_desp"] = True
    if "submit_desp" in st.session_state:
        show_extracted_features(selected_anyou, selected_suqiu_list, user_input)

        st.subheader("进行提问")
        question_answers = {}
        factor_sentence_list = {}
        count = 1
        repeated_question_management = None
        show_next_qa(selected_anyou, selected_suqiu_list, user_input, question_answers, factor_sentence_list, count, repeated_question_management)
