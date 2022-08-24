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

from RelevantLaws.LegalLibrary.relevant_laws_search import get_law_search_result
from Utils.io import read_json_attribute_value
from Utils.http_response import response_successful_result

app = Flask(__name__)


@app.route('/get_filter_conditions_of_law', methods=["get"])
def get_filter_conditions():
    filer_conditions = read_json_attribute_value("RelevantLaws/api/filter_conditions.json", "filter_conditions")
    return response_successful_result(filer_conditions)


def _construct_result_format(search_result) -> List:
    result = []
    for index, row in search_result.iterrows():
        if row['isValid'] == '有效':
            row['isValid'] = '现行有效'
        if row['locality'] == '':
            row['locality'] = '全国'
        result.append({"law_name": row['title'], "law_type": row['source'],
                            "timeliness": row['isValid'], "using_range": row['locality'],
                            "law_chapter": row['resultChapter'],
                            "law_item": row['resultSection'], "law_content": row['resultClause']})
    return result


def _get_search_result(query, filter_conditions):
    search_result = get_law_search_result(query, filter_conditions.get("timeliness"), filter_conditions.get("types_of_law"), filter_conditions.get("size", 10))
    return _construct_result_format(search_result)


@app.route('/search_laws', methods=["post"])
def search_laws():
    query = request.json.get("query")
    filter_conditions = request.json.get("filter_conditions")
    return response_successful_result(_get_search_result(query, filter_conditions))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8135, debug=True)
