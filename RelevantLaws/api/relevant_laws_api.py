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

from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.applicable_law_dto import ApplicableLawDTO
from RelevantLaws.LegalLibrary.relevant_laws_search import get_law_search_result
from Utils.io import read_json_attribute_value
from Utils.http_response import response_successful_result
from RelevantLaws.repository import relevant_laws_repository as repository
app = Flask(__name__)


@app.route('/get_filter_conditions_of_law', methods=["get"])
def get_filter_conditions():
    filer_conditions = read_json_attribute_value("RelevantLaws/api/filter_conditions.json", "filter_conditions")
    return response_successful_result(filer_conditions)


def _get_law_table_name(law_type):
    mapping = {
        '法律': 'flfg_result_falv',
        '行政法规': 'flfg_result_xzfg',
        # '监察法规': 2,
        '司法解释': 'flfg_result_sfjs',
        '宪法': 'flfg_result_xf',
        '地方性法规': 'flfg_result_dfxfg'
    }
    return mapping.get(law_type, "none")


def _construct_result_format(search_result) -> List:
    result = []
    for index, row in search_result.iterrows():
        if row['isValid'] == '有效':
            row['isValid'] = '现行有效'
        if row['locality'] == '':
            row['locality'] = '全国'
        # TODO:这里的判断有点简单了。目的是当law_item为空字符串时，把内容填上。需要修改。
        if row['resultSection'] == "" and str(row['resultClause']).startswith("第"):
            row['resultSection'] = str(row['resultClause']).split(":")[0]

        result.append({
            "law_id": _get_law_table_name(row['source']) + "#" + row['md5Clause'],
            "law_name": row['title'],
            "law_type": row['source'],
            "timeliness": row['isValid'],
            "using_range": row['locality'],
            "law_chapter": row['resultChapter'],
            "law_item": row['resultSection'],
            "law_content": row['resultClause']})
    return result


def _get_search_result(query, filter_conditions):
    search_result = get_law_search_result(query, filter_conditions.get("timeliness"), filter_conditions.get("types_of_law"), filter_conditions.get("size", 10))
    return _construct_result_format(search_result)


@app.route('/search_laws', methods=["post"])
def search_laws():
    query = request.json.get("query")
    filter_conditions = request.json.get("filter_conditions", dict())
    result = _get_search_result(query, filter_conditions)
    # TODO: 用实现于分页的total amount
    return response_successful_result(result, {"total_amount": len(result)})


@app.route('/get_law_by_law_id', methods=["get"])
def get_law_by_law_id():
    raw_law_id = request.args.get("law_id")
    table_name = str(raw_law_id).split("#")[0]
    law_id = str(raw_law_id).split("#")[1]
    return response_successful_result(repository.get_law_by_law_id(law_id, table_name))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8135, debug=True)
