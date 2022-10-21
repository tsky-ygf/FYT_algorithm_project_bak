#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  :
@Time    : 2022/8/10 13:14 
@Desc    : None
"""
import json

from flask import Flask, request

from Utils.http_response import response_successful_result, response_failed_result
from ProfessionalSearch.src.similar_case_retrival.process_by_es.cases_search import (
    get_case_search_result,
)

# from ProfessionalSearch.SimilarCaseRetrieval.src.narrative_similarity_predict import predict_fn as predict_fn_similar_cases
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
    pass


@app.route("/get_filter_conditions_of_case", methods=["get"])
def get_filter_conditions_of_case():
    try:
        filter_conditions = read_json_attribute_value(
            "ProfessionalSearch/config/similar_case_retrival/filter_conditions.json",
            "filter_conditions",
        )
        return filter_conditions
    except Exception as e:
        return response_failed_result("error:" + repr(e))


def _get_case_result(query, filter_conditions, page_num, page_size):
    if isinstance(filter_conditions, dict):
        search_result, total_num = get_case_search_result(
            query,
            filter_conditions.get("type_of_case"),
            filter_conditions.get("court_level"),
            filter_conditions.get("type_of_document"),
            filter_conditions.get("region"),
            page_num,
            page_size,
        )
    else:
        search_result, total_num = get_case_search_result(
            query,
            filter_conditions.type_of_case,
            filter_conditions.court_level,
            filter_conditions.type_of_document,
            filter_conditions.region,
            page_num,
            page_size,
        )
    result, total_num = _construct_result_format(search_result, total_num)
    if total_num >= 200:
        return {"cases": result, "total_amount": 200}
    else:
        return {"cases": result, "total_amount": len(result)}


@app.route("/search_cases", methods=["post"])
def search_cases():
    try:
        input_json = request.get_data()
        if input_json is not None:
            input_dict = json.loads(input_json.decode("utf-8"))
            query = input_dict["query"]
            filter_conditions = input_dict["filter_conditions"]
            page_number = input_dict["page_number"]
            page_size = input_dict["page_size"]
            if query is not None and filter_conditions is not None:
                result = _get_case_result(
                    query, filter_conditions, page_number, page_size
                )
                # 返回数量，若200以上，则返回200，若小于200，则返回实际number
                return result
            else:
                return response_successful_result([], {"total_amount": len([])})
            # TODO: 实现用于分页的total_amount
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)
    except Exception as e:
        return response_failed_result("error:" + repr(e))



def containenglish(str0):
    import re

    return bool(re.search("[a-zA-Z]", str0))


def _construct_result_format(search_result, total_num):
    result = []
    for index, row in search_result.iterrows():
        if (
            not row["faYuan_name"]
            or not row["event_num"]
            or containenglish(row["faYuan_name"])
            or containenglish(row["event_num"])
        ):
            total_num -= 1
            continue
        if row["table_name"] == "judgment_minshi_data_cc":
            row["table_name"] = "judgment_minshi_data"
        result.append(
            {
                "doc_id": row["table_name"] + "_SEP_" + row["uq_id"],
                "court": row["faYuan_name"],
                "case_number": row["event_num"],
                "jfType": row["jfType"],
                "content": row["content"],
            }
        )
    return result, total_num


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8140, debug=True)
