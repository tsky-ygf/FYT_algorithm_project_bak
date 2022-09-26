#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/10 13:14 
@Desc    : None
"""
import json

from flask import Flask, request

from LawsuitPrejudgment.lawsuit_prejudgment.core import civil_similar_case
from Utils.http_response import response_successful_result
from ProfessionalSearch.SimilarCaseRetrieval.core import similar_case_retrieval_service as service
from ProfessionalSearch.SimilarCaseRetrieval.core.relevant_cases_search import get_case_search_result
from typing import List

from Utils.io import read_json_attribute_value

app = Flask(__name__)


def _preprocess(request_json):
    problem = request_json.get("problem", "")
    if problem is None or str(problem).strip() == "":
        problem = ""

    claim_list = request_json.get("claim_list", [])
    if claim_list is None or len(claim_list) == 0:
        claim_list = []

    fact = request_json.get("fact")
    return problem, claim_list, fact


@app.route("/get_similar_cases", methods=["post"])
def get_similar_cases():
    problem, claim_list, fact = _preprocess(request.json)
    return response_successful_result(service.get_similar_cases(problem, claim_list, fact))


@app.route('/get_filter_conditions_of_case', methods=["get"])
def get_filter_conditions_of_case():
    filter_conditions = {
        "type_of_case": {
            "name": "案件类型",
            "is_multiple_choice": True,
            "value": [
                "全部",
                "刑事",
                "民事",
                "行政",
                "执行"  # 刑事、民事、行政、执行、其他
            ]
        },
        # "type_of_anyou":{
        #     "xingshi":{
        #         "name":"刑事",
        #         "is_multiple_choice":True,
        #         "value":[
        #
        #         ]
        #     },
        #     "mingshi": {
        #         "name": "民事",
        #         "is_multiple_choice": True,
        #         "value": [
        #
        #         ]
        #     },
        #     "xingzhen": {
        #         "name": "行政",
        #         "is_multiple_choice": True,
        #         "value": [
        #
        #         ]
        #     },
        #     "zhixing": {
        #         "name": "执行",
        #         "is_multiple_choice": True,
        #         "value": [
        #
        #         ]
        #     },
        # },
        "court_level": {
            "name": "法院层级",
            "is_multiple_choice": True,
            "value": [
                "全部",
                "最高",
                "高级",
                "中级",
                "基层"
            ]
        },
        "type_of_document": {
            "name": "文书类型",
            "is_multiple_choice": True,
            "value": [
                "全部",
                "判决",
                "裁定",
                "调解"
            ]
        },
        "region": {
            "name": "地域",
            "is_multiple_choice": True,
            "value": [
                '全国',
                '安徽省', '北京市', '重庆市', '福建省', '甘肃省', '广东省', '广西壮族自治区', '贵州省', '海南省',
                '河北省', '河南省', '黑龙江省',
                '湖北省', '湖南省', '吉林省', '江苏省', '江西省', '辽宁省', '内蒙古自治区', '宁夏回族自治区', '青海省',
                '山东省', '山西省', '陕西省', '上海市', '四川省',
                '天津市', '西藏自治区', '新疆维吾尔自治区', '云南省', '浙江省'
            ]
        }
    }
    return response_successful_result(filter_conditions)


def _get_search_result(query, filter_conditions, page_num, page_size):
    search_result = get_case_search_result(query,
                                           filter_conditions.get("type_of_case"),
                                           filter_conditions.get("court_level"),
                                           filter_conditions.get("type_of_document"),
                                           filter_conditions.get("region"),
                                           page_num,
                                           page_size)
    return _construct_result_format(search_result)


@app.route('/search_cases', methods=["post"])
def search_cases():
    # query = request.json.get("query")
    # filter_conditions = request.json.get("filter_conditions")
    # result = _get_search_result(query, filter_conditions)
    # try:
    input_json = request.get_data()
    if input_json is not None:
        input_dict = json.loads(input_json.decode("utf-8"))
        query = input_dict['query']
        filter_conditions = input_dict['filter_conditions']
        page_number = input_dict['page_number']
        page_size = input_dict['page_size']
        if query is not None and filter_conditions is not None:
            result = _get_search_result(query, filter_conditions, page_number, page_size)
            return response_successful_result(result, {"total_amount": len(result)})
        else:
            return response_successful_result([], {"total_amount": len([])})
        # TODO: 实现用于分页的total_amount
    else:
        return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    # except Exception as e:
    #     logging.info(traceback.format_exc())
    #     return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


@app.route('/get_law_document', methods=["get"])
def get_law_document():
    # doc_id = request.args.get("doc_id")
    input_json = request.get_data()
    if input_json is not None:
        input_dict = json.loads(input_json.decode("utf-8"))
        doc_id = input_dict['doc_id']
        result = service.get_criminal_law_document(doc_id)
        if result:
            return response_successful_result(result)

        law_documents = civil_similar_case.get_civil_law_documents_by_id_list([doc_id])
        if law_documents:
            result = {
                "doc_id": law_documents[0]["doc_id"],
                "html_content": law_documents[0]["raw_content"]
            }
        else:
            result = None
        return response_successful_result(result)


def _construct_result_format(search_result) -> List:
    result = []
    for index, row in search_result.iterrows():
        result.append({
            "doc_id": row['table_name'] + '_SEP_' + row['uq_id'],
            "court": row['faYuan_name'],
            "case_number": row['event_num'],
            "jfType": row['jfType'],
            "content": row['content']})
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8160, debug=True)
