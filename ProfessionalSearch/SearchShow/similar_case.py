import requests
import streamlit as st
from Utils.logger import Logger


logger = Logger(name="case-lib", level="debug").logger


def similar_case_review():
    url_case_conditions = "http://172.19.82.199:8160/get_filter_conditions_of_case"
    resp_case_json = requests.get(url_case_conditions).json()
    case_type = st.sidebar.selectbox(
        "请选择案件类型",
        resp_case_json["result"].get("type_of_case").get("value"),
        key="case_type",
    )

    court_level = st.sidebar.selectbox(
        "请选择法院层级",
        resp_case_json["result"].get("court_level").get("value"),
        key="court_level",
    )

    document_type = st.sidebar.selectbox(
        "请选择文书类型",
        resp_case_json["result"].get("type_of_document").get("value"),
        key="document_type",
    )

    prov_type = st.sidebar.selectbox(
        "请选择地域", resp_case_json["result"].get("region").get("value"), key="prov_type",
    )
    size = st.sidebar.number_input(
        "搜索结果展示条数", value=10, key="size", min_value=1, max_value=100
    )

    text = st.text_input("请输入案例内容", value="", key="text")

    run = st.button("查询", key="run")
    if run:
        url_search_case = "http://172.19.82.199:8160/search_cases"
        query = text
        filter_conditions = {
            "type_of_case": [case_type],
            "court_level": [court_level],
            "type_of_document": [document_type],
            "region": [prov_type],
        }
        input_json = {
            "page_number": 1,
            "page_size": size,
            "query": query,
            "filter_conditions": filter_conditions,  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
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
