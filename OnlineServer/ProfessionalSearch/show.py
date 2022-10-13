#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : show.py
# @Software: PyCharm
import requests
import streamlit as st
import streamlit.components.v1 as components
# from ProfessionalSearch.src.similar_case_retrival.similar_case.util import get_civil_law_documents_by_id_list
from Utils.logger import Logger


class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps_search = []

    def add_app_search(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps_search.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.radio(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'])
        app['function']()

def search():

    case, law = st.tabs(["案例检索", "法条检索"])
    with case:
        similar_case_review()
    with law:
        relevant_laws_review()
    # app = MultiApp()
    # app.add_app_search("案例检索测试服务", similar_case_review)
    # app.add_app_search("法条检索测试服务", relevant_laws_review)
    # app_search = st.sidebar.selectbox(
    #     '请选择检索测试服务类型',
    #     app.apps_search,
    #     format_func=lambda app_search: app_search['title'])
    # app_search['function']()

def similar_case_review():
    logger = Logger(name="case-lib", level="debug").logger
    url_case_conditions = 'http://127.0.0.1:8140/get_filter_conditions_of_case'
    resp_case_json = requests.get(url_case_conditions).json()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        case_type = st.selectbox(
            "请选择案件类型", resp_case_json["result"].get("type_of_case").get("value"), key="case_type"
        )
    with col2:
        court_level = st.selectbox(
            "请选择法院层级", resp_case_json["result"].get("court_level").get("value"), key="court_level"
        )
    with col3:
        document_type = st.selectbox(
            "请选择文书类型", resp_case_json["result"].get("type_of_document").get("value"), key="document_type"
        )
    with col4:
        prov_type = st.selectbox(
            "请选择地域",
            resp_case_json["result"].get("region").get("value"),
            key="prov_type",
        )
    with col5:
        size = st.number_input(
            "搜索结果展示条数", value=10, key="size", min_value=1, max_value=100
        )

    text = st.text_input("请输入案例内容", value="", key="text")

    run = st.button("查询", key="run")
    if run:
        url_search_case = 'http://127.0.0.1:8140/search_cases'
        query = text
        filter_conditions = {
            'type_of_case': [case_type],
            'court_level': [court_level],
            'type_of_document': [document_type],
            'region': [prov_type],
        }
        input_json = {
            "page_number": 1,
            "page_size": size,
            "query": query
            , "filter_conditions": filter_conditions  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
        }

        resp_json = requests.post(url_search_case, json=input_json).json()

        for index, row in enumerate(resp_json.get("result")):
            logger.info(row)
            if row["doc_id"].split("_SEP_")[0] == "judgment_minshi_data":
                row["table_name"] = "民事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_minshi_data_cc":
                row["table_name"] = "民事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_xingshi_data":
                row["table_name"] = "刑事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_zhixing_data":
                row["table_name"] = "执行"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_xingzheng_data":
                row["table_name"] = "行政"
            res_dict = {
                "标号": index,
                "案件类型": row["table_name"],
                "法院名字": row["court"],
                "文书号": row["case_number"],
                "案由": row["jfType"],
                "内容": row["content"],
            }
            st.write(res_dict)
            st.write("-" * 20 + "我是分割线" + "-" * 20)

def similar_case_review():
    logger = Logger(name="case-lib", level="debug").logger
    url_case_conditions = 'http://127.0.0.1:8140/get_filter_conditions_of_case'
    resp_case_json = requests.get(url_case_conditions).json()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        case_type = st.selectbox(
            "请选择案件类型", resp_case_json["result"].get("type_of_case").get("value"), key="case_type"
        )
    with col2:
        court_level = st.selectbox(
            "请选择法院层级", resp_case_json["result"].get("court_level").get("value"), key="court_level"
        )
    with col3:
        document_type = st.selectbox(
            "请选择文书类型", resp_case_json["result"].get("type_of_document").get("value"), key="document_type"
        )
    with col4:
        prov_type = st.selectbox(
            "请选择地域",
            resp_case_json["result"].get("region").get("value"),
            key="prov_type",
        )
    with col5:
        size = st.number_input(
            "搜索结果展示条数", value=10, key="size", min_value=1, max_value=100
        )

    text = st.text_input("请输入案例内容", value="", key="text")

    run = st.button("查询", key="run")
    if run:
        url_search_case = 'http://127.0.0.1:8140/search_cases'
        query = text
        filter_conditions = {
            'type_of_case': [case_type],
            'court_level': [court_level],
            'type_of_document': [document_type],
            'region': [prov_type],
        }
        input_json = {
            "page_number": 1,
            "page_size": size,
            "query": query
            , "filter_conditions": filter_conditions  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
        }

        resp_json = requests.post(url_search_case, json=input_json).json()

        for index, row in enumerate(resp_json.get("result")):
            logger.info(row)
            if row["doc_id"].split("_SEP_")[0] == "judgment_minshi_data":
                row["table_name"] = "民事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_minshi_data_cc":
                row["table_name"] = "民事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_xingshi_data":
                row["table_name"] = "刑事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_zhixing_data":
                row["table_name"] = "执行"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_xingzheng_data":
                row["table_name"] = "行政"
            res_dict = {
                "标号": index,
                "案件类型": row["table_name"],
                "法院名字": row["court"],
                "文书号": row["case_number"],
                "案由": row["jfType"],
                "内容": row["content"],
            }
            st.write(res_dict)
            st.write("-" * 20 + "我是分割线" + "-" * 20)

def similar_case_side_review():
    logger = Logger(name="case-lib", level="debug").logger
    url_case_conditions = 'http://127.0.0.1:8140/get_filter_conditions_of_case'
    resp_case_json = requests.get(url_case_conditions).json()
    case_type = st.sidebar.selectbox(
        "请选择案件类型", resp_case_json["result"].get("type_of_case").get("value"), key="case_type"
    )
    court_level = st.sidebar.selectbox(
        "请选择法院层级", resp_case_json["result"].get("court_level").get("value"), key="court_level"
    )
    document_type = st.sidebar.selectbox(
        "请选择文书类型", resp_case_json["result"].get("type_of_document").get("value"), key="document_type"
    )
    prov_type = st.sidebar.selectbox(
        "请选择地域",
        resp_case_json["result"].get("region").get("value"),
        key="prov_type",
    )
    size = st.sidebar.number_input(
        "搜索结果展示条数", value=10, key="size", min_value=1, max_value=100
    )

    text = st.text_input("请输入案例内容", value="", key="text")

    run = st.button("查询", key="run")
    if run:
        url_search_case = 'http://127.0.0.1:8140/search_cases'
        query = text
        filter_conditions = {
            'type_of_case': [case_type],
            'court_level': [court_level],
            'type_of_document': [document_type],
            'region': [prov_type],
        }
        input_json = {
            "page_number": 1,
            "page_size": size,
            "query": query
            , "filter_conditions": filter_conditions  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
        }

        resp_json = requests.post(url_search_case, json=input_json).json()

        for index, row in enumerate(resp_json.get("result")):
            logger.info(row)
            if row["doc_id"].split("_SEP_")[0] == "judgment_minshi_data":
                row["table_name"] = "民事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_minshi_data_cc":
                row["table_name"] = "民事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_xingshi_data":
                row["table_name"] = "刑事"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_zhixing_data":
                row["table_name"] = "执行"
            elif row["doc_id"].split("_SEP_")[0] == "judgment_xingzheng_data":
                row["table_name"] = "行政"
            res_dict = {
                "标号": index,
                "案件类型": row["table_name"],
                "法院名字": row["court"],
                "文书号": row["case_number"],
                "案由": row["jfType"],
                "内容": row["content"],
            }
            st.write(res_dict)
            st.write("-" * 20 + "我是分割线" + "-" * 20)


def relevant_laws_review():
    logger = Logger(name="law-lib", level="debug").logger
    url_law_conditions = "http://127.0.0.1:8139/get_filter_conditions_of_law"
    resp_conditions_json = requests.get(url_law_conditions).json()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        legal_type = st.selectbox(
            "请选择法条种类", resp_conditions_json["result"].get("types_of_law").get("value"), key="legal_type"
        )
    with col2:
        isValid_type = st.selectbox(
            "请选择法律时效性", resp_conditions_json["result"].get("timeliness").get("value"), key="text"
        )
    with col3:
        prov_type = st.selectbox(
            "请选择使用范围",
            resp_conditions_json["result"].get("scope_of_use").get("value"),
            key="text",
        )
    with col4:
        size = st.number_input(
            "搜索结果展示条数", value=10, key="size_law", min_value=1, max_value=100
        )






    text = st.text_input("请输入法条内容", value="", key="text")
    run = st.button("查询", key="run_law")

    if run:
        url = "http://127.0.0.1:8139/search_laws"
        body = {
            "query": text,
            "filter_conditions": {
                "types_of_law": [
                    legal_type
                ],
                "timeliness": [
                    isValid_type
                ],
                "scope_of_use": [
                    prov_type
                ]
            },
            "page_number": 1,
            "page_size": size
        }
        resp_json = requests.post(url, json=body).json()

        for index, row in enumerate(resp_json.get("result")):
            logger.info(row)
            res_dict = {
                "标号": index,
                # "md5": row["md5Clause"],
                "标题": row["law_name"],
                "法律类别": row["law_type"],
                "时效性": row["timeliness"],
                "使用范围": row["using_range"],
                "法条章节": row["law_chapter"],
                "法条条目": row["law_item"],
                "法条内容": row["law_content"],
            }

            st.write(res_dict)
            st.write("-" * 20 + "我是分割线" + "-" * 20)


def relevant_laws_side_review():
    logger = Logger(name="law-lib", level="debug").logger
    url_law_conditions = "http://127.0.0.1:8139/get_filter_conditions_of_law"
    resp_conditions_json = requests.get(url_law_conditions).json()
    legal_type = st.sidebar.selectbox(
        "请选择法条种类", resp_conditions_json["result"].get("types_of_law").get("value"), key="legal_type"
    )
    isValid_type = st.sidebar.selectbox(
        "请选择法律时效性", resp_conditions_json["result"].get("timeliness").get("value"), key="text"
    )
    prov_type = st.sidebar.selectbox(
        "请选择使用范围",
        resp_conditions_json["result"].get("scope_of_use").get("value"),
        key="text",
    )
    size = st.sidebar.number_input(
        "搜索结果展示条数", value=10, key="size_law", min_value=1, max_value=100
    )
    text = st.text_input("请输入法条内容", value="", key="text")
    run = st.button("查询", key="run_law")

    if run:
        url = "http://127.0.0.1:8139/search_laws"
        body = {
            "query": text,
            "filter_conditions": {
                "types_of_law": [
                    legal_type
                ],
                "timeliness": [
                    isValid_type
                ],
                "scope_of_use": [
                    prov_type
                ]
            },
            "page_number": 1,
            "page_size": size
        }
        resp_json = requests.post(url, json=body).json()

        for index, row in enumerate(resp_json.get("result")):
            logger.info(row)
            res_dict = {
                "标号": index,
                # "md5": row["md5Clause"],
                "标题": row["law_name"],
                "法律类别": row["law_type"],
                "时效性": row["timeliness"],
                "使用范围": row["using_range"],
                "法条章节": row["law_chapter"],
                "法条条目": row["law_item"],
                "法条内容": row["law_content"],
            }

            st.write(res_dict)
            st.write("-" * 20 + "我是分割线" + "-" * 20)

import logging
from typing import List, Dict
import pymysql

def get_civil_law_documents_by_id_list(id_list: List[str], table_name) -> List[Dict]:
    # 打开数据库连接
    db = pymysql.connect(host='172.19.82.227',
                         user='root',
                         password='Nblh@2022',
                         database='judgments_data')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    try:
        format_strings = ','.join(['%s'] * len(id_list))
        # 执行SQL语句
        cursor.execute("SELECT uq_id, jslcm, source_txt,yg_sc,bg_sc, byrw, pubDate FROM " + table_name + " WHERE uq_id in (%s)" % format_strings,
                       tuple(id_list))
        # 获取所有记录列表
        fetched_data = cursor.fetchall()
        law_documents = [{
            "uq_id": row[0],
            "jslcm": row[1],
            "source_txt": row[2],
            "yg_sc": row[3],
            "bg_sc": row[4],
            "byrw": row[5],
            "pubDate": row[6]
        } for row in fetched_data]
    except:
        logging.error("Error: unable to fetch data")
        law_documents = []
    # 关闭数据库连接
    db.close()
    return law_documents

def similar_case_retrieval_review():
    # 初始化
    logger = Logger(name="law-lib", level="debug").logger
    url_similar_case = "http://127.0.0.1:8163/top_k_similar_narrative"
    url_jfType = "http://101.69.229.138:7100/get_civil_problem_summary"
    url_suqiu = "http://101.69.229.138:7100/get_claim_list_by_problem_id"
    jfType_list = requests.get(url_jfType).json()
    suqiu_req = {"problem_id": "", "fact": ""}
    suqiu_res = requests.post(url_suqiu, json=suqiu_req).json()
    # 获取相似案例
    groupName_list = []
    for value_item in jfType_list["value"]:
        groupName_list.append(value_item["groupName"])
    groupName = st.sidebar.selectbox(
        "请选择一级纠纷类型", groupName_list, key="legal_type"
    )
    groupList, problem_list = [], []
    for value_item in jfType_list["value"]:
        if groupName == value_item["groupName"]:
            groupList = value_item["groupList"]
    for problem_item in groupList:
        problem_list.append(problem_item['problem'])
    jfType = st.sidebar.selectbox(
        "请选择二级纠纷类型", problem_list, key="text"
    )
    fact = st.text_input("请输入事实内容", value="", key="text")
    similar_case_req = {"problem": jfType, "claim_list": [], "fact": fact}
    run = st.button("查询", key="run_law")

    if run:
        suqiu_res = requests.post(url_similar_case, json=similar_case_req).json()
        print(suqiu_res)
        # 组织结果返回
        doc_id_list, sim_list, reason_name_list, tags_list = suqiu_res["dids"], suqiu_res["sims"], suqiu_res["reasonNames"], suqiu_res["tags"]
        detail_link = "http://101.69.229.138:7145/get_law_document?doc_id=judgment_minshi_data_SEP_"
        if doc_id_list:
            jslcm_list = get_civil_law_documents_by_id_list(doc_id_list, "judgment_minshi_data")
            for index, uq_id in enumerate(doc_id_list):
                logger.info(reason_name_list[index])
                jslcm, detail, yg_sc, bg_sc, byrw, date = "", "", "", "", "", ""
                for jslcm_item in jslcm_list:
                    if uq_id == jslcm_item['uq_id']:
                        jslcm = jslcm_item['jslcm']
                        detail = jslcm_item['source_txt']
                        date = jslcm_item['pubDate']
                        # yg_sc = jslcm_item['yg_sc']
                        # bg_sc = jslcm_item['bg_sc']
                        # byrw = jslcm_item['byrw']
                        link = "[link]"+"(" + detail_link + uq_id+")"
                res_dict = {
                    "标号": index,
                    "唯一ID": uq_id,
                    # "原告诉称": yg_sc,
                    # "被告诉称": bg_sc,
                    # "本院认为": byrw,
                    "经审理查明": jslcm,
                    "相似率": sim_list[index],
                    "纠纷类型": reason_name_list[index],
                    "关键词": tags_list[index],
                    "时间": date
                    # "裁判文书详情": detail
                }

                st.write(res_dict)
                components.html(detail, scrolling=True)
                # st.write("裁判文书详情", link)
                st.write("-" * 20 + "我是分割线" + "-" * 20)


def _tabs(tabs_data={}, default_active_tab=0):
    tab_titles = list(tabs_data.keys())
    if not tab_titles:
        return None
    active_tab = st.radio("", tab_titles, index=default_active_tab, horizontal=True)
    return tabs_data[active_tab]


def do_tabs():
    tab_content = _tabs({
        "案例检索": similar_case_side_review,
        "法条检索": relevant_laws_side_review,
        "相似案例检索": similar_case_retrieval_review,
    })
    if callable(tab_content):
        tab_content()
    elif type(tab_content) == str:
        st.markdown(tab_content, unsafe_allow_html=True)
    else:
        st.write(tab_content)
