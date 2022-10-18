#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 15:28 
@Desc    : None
"""
import pandas as pd
import requests
import streamlit as st

URL = "http://127.0.0.1:8105"

""" 行政预判 """


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
    supported_administrative_list = requests.get(url=URL + "/get_administrative_type").json().get("result")
    selected_type_name = st.selectbox("请选择你遇到的纠纷类型", [item["type_name"] for item in supported_administrative_list])
    type_id = next(
        (item["type_id"] for item in supported_administrative_list if item["type_name"] == selected_type_name), None)

    problem_and_situation_list = requests.get(url=URL + "/get_administrative_problem_and_situation_by_type_id",
                                              params={"type_id": type_id}).json().get("result")
    selected_problem = st.selectbox("请选择你遇到的问题", [item["problem"] for item in problem_and_situation_list], key="一级")
    selected_situation = st.selectbox("请选择具体的情形", next(
        (item["situations"] for item in problem_and_situation_list if item["problem"] == selected_problem), None),
                                      key="情形")

    run = st.button("开始计算", key="run")
    if run:
        _show_administrative_report(type_id, selected_situation)


""" 刑事预判 """


def get_criminal_result(fact, question_answers, factor_sentence_list, anyou, event):
    body = {
        "fact": fact,
        "question_answers": question_answers
        # "factor_sentence_list": factor_sentence_list,
        # "anyou": anyou,
        # "event": event
    }
    resp_json = requests.post(url="http://127.0.0.1:8105/get_criminal_result", json=body).json()
    if resp_json.get("success") is False:
        raise Exception("刑事预判接口返还异常: {}".format(resp_json.get("error_msg")))

    next_question_info = resp_json.get("question_next")
    if next_question_info:
        resp_json["next_question"] = str(next_question_info).split(":")[0]
        resp_json["answers"] = str(next_question_info).split(":")[1].split(";")
        resp_json["single_or_multi"] = "single" if resp_json.get("question_type") == "1" else "multi"
        return True, resp_json
    return False, resp_json


def _remove_criminal_click_state(_count):
    while True:
        key = 'criminal_ask_btn' + str(_count) + '_clicked'
        if key in st.session_state:
            del st.session_state[key]
            _count += 1
            print("remove key:", key)
        else:
            break
    pass


def show_criminal_next_qa(_user_input, _question_answers, _factor_sentence_list, _anyou, _event, _count):
    has_next_question, info = get_criminal_result(_user_input, _question_answers, _factor_sentence_list, _anyou,
                                                          _event)
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
                st.radio("", options=answers, key='criminal_aks_' + str(_count), on_change=_remove_criminal_click_state,
                         args=(_count,))]
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
            _factor_sentence_list = next_question_info.get('factor_sentence_list')
            _anyou = next_question_info.get('anyou')
            _event = next_question_info.get('event')
            _count += 1
            show_criminal_next_qa(_user_input, _question_answers, _factor_sentence_list, _anyou, _event, _count)
    else:
        show_criminal_report(info.get("result"))
    pass


def show_criminal_report(result):
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
        show_criminal_next_qa(user_input, question_answers, factor_sentence_list, anyou, event, count)


""" 民事预判 """


def get_anyou_list():
    # return ["借贷纠纷", "劳动社保", "买卖合同", "租赁合同"]
    resp_json = requests.get(URL + "/get_civil_problem_summary").json()
    return [problem_info for group_info in resp_json["value"] for problem_info in group_info["groupList"]]


def get_suqiu_list(problem_id, fact):
    # return ["支付劳动劳务报酬", "支付加班工资", "支付双倍工资", "经济补偿金或赔偿金", "劳务受损赔偿", "劳动劳务致损赔偿"]
    resp_json = requests.post(URL + "/get_claim_list_by_problem_id", json={"problem_id": problem_id, "fact": fact}).json()
    return resp_json["value"]


def _request(problem, claim_list, fact, question_answers, factor_sentence_list, repeated_question_management):
    url = URL + "/reasoning_graph_result"
    body = {
        "problem": problem,
        "claim_list": claim_list,
        "fact": fact,
        "question_answers": question_answers,
        "factor_sentence_list": factor_sentence_list
    }
    return requests.post(url, json=body).json()


def get_extracted_features(anyou, suqiu_list, desp):
    # extracted_features = {
    #     "特征": ['无法偿还贷款', '存在借款合同', '不存在借款合同', '出借方未实际提供借款', '借款人逾期未返还借款'],
    #     "对应句子": ['现在公司没有按时还款', '我要求公司按照合同约定', '我要求公司按照合同约定', '返还借款本金130万及利息、违约金、律师费66800元', '现在公司没有按时还款'],
    #     "正向/负向匹配": [1, 1, -1, -1, 1],
    #     "匹配表达式": ['(((没有|没|未|不|非|无|未经|怠于)[^。；，：,;:？！!?\s]*(偿还|归还|偿付|清偿|还款|还清|还本付息|偿清|还债|还账|还钱|付清|结清|返还|支付)))',
    #               '((((签|写|打)[^。；，：,;:？！!?\s]*(借据|借条|欠条|合同|协议)|借据|借条|欠条|合同|协议)))',
    #               '((((签|写|打)[^。；，：,;:？！!?\s]*(借据|借条|欠条|合同|协议)|借据|借条|欠条|合同|协议)))',
    #               '(((给|提供|付|借)[^。；，：,;:？！!?\s]*(款|钱|资金)))',
    #               '(((没有|没|未|不|非|无|未经|怠于)[^。；，：,;:？！!?\s]*(偿还|归还|偿付|清偿|还款|还清|还本付息|偿清|还债|还账|还钱|付清|结清|返还|支付)))']
    # }
    resp = _request(anyou, suqiu_list, desp, {}, [], None)
    factor_sentence_list = resp.get("factor_sentence_list", [])
    extracted_features = {
        "特征": [],
        "对应句子": [],
        "正向/负向匹配": [],
        "匹配表达式": []
    }

    for sentence, feature, flag, expression in factor_sentence_list:
        extracted_features["特征"].append(feature)
        extracted_features["对应句子"].append(sentence)
        extracted_features["正向/负向匹配"].append(flag)
        extracted_features["匹配表达式"].append(expression)
    return extracted_features


def get_next_question(anyou, suqiu_list, desp, question_answers, factor_sentence_list, repeated_question_management):
    resp = _request(anyou, suqiu_list, desp, question_answers, factor_sentence_list, repeated_question_management)
    next_question_info = resp.get("question_next")
    if next_question_info:
        next_question = str(next_question_info).split(":")[0]
        answers = str(next_question_info).split(":")[1].split(";")
        if resp.get("question_type") == "1":
            single_or_multi = "single"
        else:
            single_or_multi = "multi"
        return True, {"next_question": next_question, "answers": answers, "single_or_multi": single_or_multi,
                      "factor_sentence_list": resp.get("factor_sentence_list"),
                      "repeated_question_management": resp.get("repeated_question_management"),
                      "debug_info": resp.get("debug_info")}
    else:
        return False, {"result": resp.get("result"), "debug_info": resp.get("debug_info")}


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
    for k, v in debug_info.items():
        st.markdown(k)
        li = str(v).split('\n')
        for item in li:
            st.markdown(item)


def show_next_qa(_selected_anyou, _selected_suqiu_list, _user_input, _question_answers, _factor_sentence_list, _count, _repeated_question_management):
    has_next_question, info = get_next_question(_selected_anyou, _selected_suqiu_list, _user_input,
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


def show_report(result):
    st.subheader("产生评估报告")
    with st.expander("预判结果"):
        for report in result["report"]:
            for item in report:
                st.markdown("#### {}".format(item["title"]))
                if item["type"] == "TYPE_TEXT":
                    st.markdown(item["content"], unsafe_allow_html=True)
                elif item["type"] == "TYPE_LIST_OF_TEXT":
                    for every_content in item["content"]:
                        st.markdown(every_content, unsafe_allow_html=True)
                elif item["type"] == "TYPE_GRAPH_OF_PROB":
                    st.markdown("{}%".format(int(item["content"] * 100)))
    with st.expander("相似类案"):
        similar_cases = result["similar_case"]
        for case in similar_cases:
            st.markdown("**{}**".format(case["title"]))
            st.markdown("{}/ {}".format(case["court"], case["case_number"]))
            st.markdown("*{}*".format(case["tag"]))
    with st.expander("相关法条"):
        relevant_laws = result["applicable_law"]
        for law in relevant_laws:
            st.markdown("**{}**".format(law["law_name"]))
            st.markdown(law["law_content"])
        pass
    pass


def show_extracted_features(_selected_anyou, _selected_suqiu_list, _user_input):
    st.subheader("从用户描述提取到的特征")
    df = pd.DataFrame(data=get_extracted_features(_selected_anyou, _selected_suqiu_list, _user_input))
    st.dataframe(df)


def civil_prejudgment_testing_page():
    # selected_anyou = st.sidebar.selectbox(label="案由", options=get_anyou_list())
    st.subheader("案由")
    anyou_info_list = get_anyou_list()
    selected_anyou = st.selectbox(label="", options=(item["problem"] for item in anyou_info_list))


    st.subheader('描述经过')
    user_input = st.text_area('请描述您的纠纷经过，描述越全面评估越准确', '''''')
    st.subheader("选择诉求")
    suqiu_info_list = get_suqiu_list(next((item["id"] for item in anyou_info_list if item["problem"]==selected_anyou)), user_input)
    selected_suqiu_list = st.multiselect(
        '',
        (item["claim"] for item in suqiu_info_list),
        None)

    if st.button("提交评估"):
        st.session_state["submit_desp"] = True
    if "submit_desp" in st.session_state:
        show_extracted_features(selected_anyou, selected_suqiu_list, user_input)

        st.subheader("进行提问")
        question_answers = {}
        factor_sentence_list = []
        count = 1
        repeated_question_management = None
        show_next_qa(selected_anyou, selected_suqiu_list, user_input, question_answers, factor_sentence_list, count, repeated_question_management)
        
        
def lawsuit_prejudgment_testing_page():
    tab1, tab2, tab3 = st.tabs(["民事", "刑事", "行政"])

    with tab1:
        civil_prejudgment_testing_page()

    with tab2:
        criminal_prejudgment_testing_page()

    with tab3:
        administrative_prejudgment_testing_page()
