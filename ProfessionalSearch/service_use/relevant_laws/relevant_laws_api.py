#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/17 11:53 
@Desc    : 法条检索模块的接口
"""
from typing import List

from flask import Flask
from flask import request

from LawsuitPrejudgment.lawsuit_prejudgment.core.civil_relevant_law import (
    CivilRelevantLaw,
)
from ProfessionalSearch.src.relevant_laws.process_by_es.laws_search import (
    get_law_search_result,
)
from Utils.io import read_json_attribute_value
from Utils.http_response import response_successful_result, response_failed_result

app = Flask(__name__)


@app.route("/get_filter_conditions_of_law", methods=["get"])
def get_filter_conditions():
    try:
        filer_conditions = read_json_attribute_value(
            "ProfessionalSearch/config/relevant_laws/filter_conditions.json",
            "filter_conditions",
        )
        return response_successful_result(filer_conditions)
    except Exception as e:
        return response_failed_result("error:" + repr(e))


def _get_law_table_name(law_type):
    table_mapping = read_json_attribute_value(
        "ProfessionalSearch/config/relevant_laws/table_mapping.json", "table_mapping",
    )
    return table_mapping.get(law_type, "none")


def _construct_result_format(search_result) -> List:
    result = []
    for index, row in search_result.iterrows():
        if row["isValid"] == "有效":
            row["isValid"] = "现行有效"
        if row["prov"] == "" or row["prov"] == None:
            row["prov"] = "全国"
        # TODO:这里的判断有点简单了。目的是当law_item为空字符串时，把内容填上。需要修改。
        if row["resultSection"] == "" and str(row["resultClause"]).startswith("第"):
            row["resultSection"] = str(row["resultClause"]).split(":")[0]

        result.append(
            {
                "law_id": _get_law_table_name(row["source"])
                + "_SEP_"
                + row["md5Clause"],
                "law_name": row["title"],
                "law_type": row["source"],
                "timeliness": row["isValid"],
                "using_range": row["prov"],
                "law_chapter": row["resultChapter"],
                "law_item": row["resultSection"],
                "law_content": row["resultClause"],
            }
        )
    return result


def _get_law_result(query, filter_conditions, page_number, page_size):
    if isinstance(filter_conditions, dict):
        search_result, total_num = get_law_search_result(
            query,
            filter_conditions.get("timeliness"),
            filter_conditions.get("types_of_law"),
            filter_conditions.get("scope_of_use"),
            page_number,
            page_size,
        )
    else:
        search_result, total_num = get_law_search_result(
            query,
            filter_conditions.timeliness,
            filter_conditions.types_of_law,
            filter_conditions.scope_of_use,
            page_number,
            page_size,
        )
    result = _construct_result_format(search_result)
    if total_num >= 200:
        return response_successful_result(result, {"total_amount": 200})
    else:
        return response_successful_result(result, {"total_amount": len(result)})


@app.route("/search_laws", methods=["post"])
def search_laws():
    try:
        query = request.json.get("query")
        filter_conditions = request.json.get("filter_conditions", dict())
        page_number = request.json.get("page_number")
        page_size = request.json.get("page_size")
        result = _get_law_result(query, filter_conditions, page_number, page_size)
        return result
    except Exception as e:
        return response_failed_result("error:" + repr(e))


@app.route("/get_law_by_law_id", methods=["get"])  # 目前由后端查询，方法废弃
def get_law_by_law_id():
    try:
        raw_law_id = request.args.get("law_id")
        result = CivilRelevantLaw.get_law_in_memory(raw_law_id)
        if result:
            return response_successful_result(result)

        pair = str(raw_law_id).split("_SEP_")
        if len(pair) != 2:
            return response_successful_result(dict())

        table_name = pair[0]
        law_id = pair[1]
    except Exception as e:
        return response_failed_result("error:" + repr(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8139, debug=True)
