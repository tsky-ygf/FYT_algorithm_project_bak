#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : show.py
# @Software: PyCharm
import requests
import streamlit as st
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
    app = MultiApp()
    app.add_app_search("案例检索测试服务", similar_case_review)
    app.add_app_search("法条检索测试服务", relevant_laws_review)
    app_search = st.sidebar.selectbox(
        '请选择检索测试服务类型',
        app.apps_search,
        format_func=lambda app_search: app_search['title'])
    app_search['function']()


def similar_case_review():
    logger = Logger(name="case-lib", level="debug").logger
    url_case_conditions = 'http://127.0.0.1:8160/get_filter_conditions_of_case'
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
    # run = True
    query_list = []
    if run:
        url_search_case = 'http://127.0.0.1:8160/search_cases'
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
            # break
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
    url_law_conditions = "http://127.0.0.1:8161/get_filter_conditions_of_law"
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
        "搜索结果展示条数", value=10, key="size", min_value=1, max_value=100
    )

    text = st.text_input("请输入法条内容", value="", key="text")
    run = st.button("查询", key="run")

    if run:
        url = "http://127.0.0.1:8161/search_laws"
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
            # break
            # if row["timeliness"] == "有效":
            #     row["timeliness"] = "现行有效"
            # if row["using_range"] == "":
            #     row["using_range"] = "全国"
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
