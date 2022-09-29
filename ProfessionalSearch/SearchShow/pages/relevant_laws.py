import requests
import streamlit as st
from Utils.logger import Logger


logger = Logger(name="law-lib", level="debug").logger


def relevant_laws_review():
    url_law_conditions = "http://172.19.82.199:8161/get_filter_conditions_of_law"
    resp_conditions_json = requests.get(url_law_conditions).json()

    legal_type = st.sidebar.selectbox(
        "请选择法条种类",
        resp_conditions_json["result"].get("types_of_law").get("value"),
        key="legal_type",
    )

    isValid_type = st.sidebar.selectbox(
        "请选择法律时效性",
        resp_conditions_json["result"].get("timeliness").get("value"),
        key="text",
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
        url = "http://172.19.82.199:8161/search_laws"
        body = {
            "query": text,
            "filter_conditions": {
                "types_of_law": [legal_type],
                "timeliness": [isValid_type],
                "scope_of_use": [prov_type],
            },
            "page_number": 1,
            "page_size": size,
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
