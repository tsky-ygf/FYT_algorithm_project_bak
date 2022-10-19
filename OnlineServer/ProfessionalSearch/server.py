#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    :
# @File    : server.py
# @Software: PyCharm
import json

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
from ProfessionalSearch.src.similar_case_retrival.similar_case.narrative_similarity_predict import (
    predict_fn as predict_fn_similar_cases,
)

from ProfessionalSearch.service_use.relevant_laws.relevant_laws_api import (
    get_filter_conditions,
    _get_law_result,
)
from ProfessionalSearch.service_use.similar_case_retrival.similar_case_retrieval_api import (
    get_filter_conditions_of_case,
    _get_case_result,
)
from ProfessionalSearch.src.similar_case_retrival.similar_case.util import desensitization
from Utils.http_response import response_successful_result, response_failed_result

app = FastAPI()


class Filter_conditions_law(BaseModel):
    timeliness: list
    types_of_law: list
    scope_of_use: list


class Search_laws_input(BaseModel):
    query: str
    filter_conditions: Union[Filter_conditions_law, None] = None
    page_number: int
    page_size: int


class Law_id(BaseModel):
    law_id: str


class Filter_conditions_case(BaseModel):
    type_of_case: list
    court_level: list
    type_of_document: list
    region: list


class Search_cases_input(BaseModel):
    query: str
    filter_conditions: Union[Filter_conditions_case, None] = None
    page_number: int
    page_size: int


class Similar_case_input(BaseModel):
    fact: str
    problem: str
    claim_list: list


class Case_id(BaseModel):
    case_id: str


@app.get("/get_filter_conditions_of_law")
def _get_filter_conditions():
    """
    返回法条检索的输入条件
    """
    return get_filter_conditions()


@app.post("/search_laws")
def search_laws(search_query: Search_laws_input):
    """
    获取法律检索的结果

    参数设定；

    @query: 用户输入的查询内容

    @filter_conditions: 用户输入的条件
        @timeliness:  用户输入的时效性
        @types_of_law： 用户输入的法律种类
        @scope_of_use： 用户输入的使用范围
    @page_number: 第几页

    @page_size: 页大小
    """
    try:
        result = _get_law_result(
            search_query.query,
            search_query.filter_conditions,
            search_query.page_number,
            search_query.page_size,
        )
    except Exception as e:
        return response_failed_result("error:" + repr(e))
    return result


@app.get("/get_law_by_law_id")
def get_law_by_law_id(law_id: Law_id):
    # TODO 待更新
    return {"result": ""}


@app.get("/get_filter_conditions_of_case")
def _get_filter_conditions():
    """
    返回案例检索的输入条件
    """
    return get_filter_conditions_of_case()


@app.post("/search_cases")
def search_laws(search_query: Search_cases_input):
    """
    获取法律检索的结果

    参数设定；

    @query: 用户输入的查询内容

    @filter_conditions: 用户输入的条件
        @type_of_case:  用户输入的案件类型
        @court_level:  用户输入的法院层级
        @type_of_document:  用户输入的文书类型
        @region:  用户输入的地域

    @page_number: 第几页

    @page_size: 页大小
    """
    try:
        if search_query is not None:
            query = search_query.query
            filter_conditions = search_query.filter_conditions
            page_number = search_query.page_number
            page_size = search_query.page_size
            if query is not None and filter_conditions is not None:
                result = _get_case_result(
                    query, filter_conditions, page_number, page_size
                )
                # 返回数量，若200以上，则返回200，若小于200，则返回实际number
                return result
            else:
                return response_successful_result([], {"total_amount": len([])})
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)
    except Exception as e:
        return response_failed_result("error:" + repr(e))


@app.get("/get_law_document")
def get_law_document(case_id: Case_id):
    # TODO 待更新
    return {"result": ""}


@app.post("/top_k_similar_narrative")
def get_similar_case(search_query: Similar_case_input):
    """
    @fact: 用户输入的事实描述
    @problem: 用户输入的纠纷类型
    @claim_list: 诉求类型
    """
    try:
        if search_query is not None:
            fact = search_query.fact
            problem = search_query.problem
            claim_list = search_query.claim_list
            (
                doc_id_list,
                sim_list,
                # win_los_list,
                reason_name_list,
                appeal_name_list,
                tags_list,
                keywords,
                pubDate_list,
            ) = predict_fn_similar_cases(fact, problem, claim_list)

            return json.dumps(
                {
                    "dids": doc_id_list,
                    "sims": sim_list,
                    # "winLos": win_los_list,
                    "reasonNames": reason_name_list,
                    "appealNames": appeal_name_list,
                    "tags": tags_list,
                    "pubDates": pubDate_list,
                    "keywords": keywords,
                    "error_msg": "",
                    "status": 0,
                },
                ensure_ascii=False,
            )

        else:
            return json.dumps(
                {"error_msg": "data is None", "status": 1}, ensure_ascii=False
            )
    except Exception as e:
        return response_failed_result("error:" + repr(e))


@app.post("/desensitization")
def get_text_desen(text: str) -> str:
    res = ''
    res = desensitization(text)
    return res


if __name__ == "__main__":
    # 日志设置
    uvicorn.run(
        "OnlineServer.ProfessionalSearch.server:app",
        host="0.0.0.0",
        port=8132,
        reload=False,
        workers=1,
    )
