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

URL = "http://127.0.0.1:8133"

""" 行政预判 """


def _show_administrative_report(result):
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


def _remove_administrative_click_state(_count):
    while True:
        key = 'administrative_ask_btn' + str(_count) + '_clicked'
        if key in st.session_state:
            del st.session_state[key]
            _count += 1
            print("remove key:", key)
        else:
            break
    pass


def _show_administrative_next_qa(dialogue_history, dialogue_state, _count):
    body = {
        "dialogue_history": dialogue_history,
        "dialogue_state": dialogue_state
    }
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    next_action = resp_json["next_action"]
    if next_action["action_type"] == "ask":
        btn_key = 'administrative_ask_btn' + str(_count)
        btn_click_key = btn_key + "_clicked"
        next_question = next_action["content"]["question"]
        answers = next_action["content"]["candidate_answers"]
        single_or_multi = next_action["content"]["question_type"]

        st.markdown('**{}**'.format(next_question))
        if single_or_multi == "single":
            selected_answers = [
                st.radio("", options=answers, key='administrative_aks_' + str(_count), on_change=_remove_administrative_click_state,
                         args=(_count,))]
        else:
            selected_answers = []
            for idx, option in enumerate(answers):
                if st.checkbox(option, key='administrative_ask_' + str(_count) + '_item_' + str(idx)):
                    selected_answers.append(option)
        st.write(selected_answers)
        if st.button("确定", key=btn_key):
            st.session_state[btn_click_key] = True
        if btn_click_key in st.session_state:
            last_question_info = next_action["content"]
            last_question_info["user_answer"] = selected_answers

            if not dialogue_history["question_answers"]:
                dialogue_history["question_answers"] = []
            dialogue_history["question_answers"].append(last_question_info)

            _show_administrative_next_qa(dialogue_history, resp_json["dialogue_state"], _count+1)
    elif next_action["action_type"] == "report":
        _show_administrative_report(next_action["content"])
    pass


def administrative_prejudgment_testing_page():
    dialogue_history = {
        "user_input": None,
        "question_answers": None
    }
    dialogue_state = {
        "domain": "administrative",
        "problem": None,
        "claim_list": None,
        "other": None
    }
    count = 1
    _show_administrative_next_qa(dialogue_history, dialogue_state, count)


""" 刑事预判 """


def get_criminal_result(fact, question_answers, factor_sentence_list, anyou, event):
    body = {
        "fact": fact,
        "question_answers": question_answers
        # "factor_sentence_list": factor_sentence_list,
        # "anyou": anyou,
        # "event": event
    }
    resp_json = requests.post(url=URL+"/get_criminal_result", json=body).json()
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


def get_extracted_features(dialogue_history, dialogue_state):
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
    body = {
        "dialogue_history": dialogue_history,
        "dialogue_state": dialogue_state
    }
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    factor_sentence_list = resp_json["dialogue_state"]["other"]["factor_sentence_list"]
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


def show_next_qa(dialogue_history, dialogue_state, _count):
    body = {
        "dialogue_history": dialogue_history,
        "dialogue_state": dialogue_state
    }
    resp_json = requests.post(URL + "/lawsuit_prejudgment", json=body).json()
    show_debug_info(resp_json["dialogue_state"]["other"]["debug_info"])

    next_action = resp_json["next_action"]
    if next_action["action_type"] == "ask":
        btn_key = 'ask_btn' + str(_count)
        btn_click_key = btn_key + "_clicked"
        next_question = next_action["content"]["question"]
        answers = next_action["content"]["candidate_answers"]
        single_or_multi = next_action["content"]["question_type"]

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
            last_question_info = next_action["content"]
            last_question_info["user_answer"] = selected_answers

            if not dialogue_history["question_answers"]:
                dialogue_history["question_answers"] = []
            dialogue_history["question_answers"].append(last_question_info)

            show_next_qa(dialogue_history, resp_json["dialogue_state"], _count+1)
    elif next_action["action_type"] == "report":
        show_report(next_action["content"])
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


def show_extracted_features(dialogue_history, dialogue_state):
    st.subheader("从用户描述提取到的特征")
    df = pd.DataFrame(data=get_extracted_features(dialogue_history, dialogue_state))
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
        dialogue_history = {
            "user_input": user_input,
            "question_answers": None
        }
        dialogue_state = {
            "domain": "civil",
            "problem": selected_anyou,
            "claim_list": selected_suqiu_list,
            "other": None
        }
        count = 1

        show_extracted_features(dialogue_history, dialogue_state)
        show_next_qa(dialogue_history, dialogue_state, count)
        
        
def lawsuit_prejudgment_testing_page():
    tab1, tab2, tab3 = st.tabs(["民事", "刑事", "行政"])

    with tab1:
        civil_prejudgment_testing_page()

    with tab2:
        criminal_prejudgment_testing_page()

    with tab3:
        administrative_prejudgment_testing_page()
