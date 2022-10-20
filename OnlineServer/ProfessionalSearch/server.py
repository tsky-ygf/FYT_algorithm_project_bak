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
from pydantic import BaseModel, Field
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
from ProfessionalSearch.src.similar_case_retrival.similar_case.util import (
    desensitization,
)
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

    class Config:
        schema_extra = {
            "example": {
                "query": "侵权",
                "filter_conditions": {
                    "types_of_law": ["地方性法规"],
                    "timeliness": ["全部"],
                    "scope_of_use": ["广东省"],
                },
                "page_number": 1,
                "page_size": 10,
            }
        }


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

    class Config:
        schema_extra = {
            "example": {
                "page_number": 1,
                "page_size": 10,
                "query": "买卖",
                "filter_conditions": {
                    "type_of_case": ["民事"],
                    "court_level": ["最高"],
                    "type_of_document": ["裁定"],
                    "region": ["安徽省"],
                },
            }
        }


class Similar_case_input(BaseModel):
    fact: str
    problem: str
    claim_list: list

    class Config:
        schema_extra = {
            "example": {{"problem": "民间借贷纠纷", "claim_list": [], "fact": "借钱不还"}}
        }


class Case_id(BaseModel):
    case_id: str


class conditions_law_result(BaseModel):
    type_of_case: dict
    court_level: dict
    type_of_document: dict
    region: dict


class search_laws_result(BaseModel):
    doc_id: str
    court: str
    case_number: str
    jfType: str
    content: str
    total_amount: int


class conditions_case_result(BaseModel):
    type_of_case: dict
    court_level: dict
    type_of_document: dict
    region: dict


class search_cases_result(BaseModel):
    doc_id: str
    court: str
    case_number: str
    jfType: str
    content: str
    total_amount: int


class similar_narrative_result(BaseModel):
    dids: list[str]
    sims: list[float]
    reasonNames: list[str]
    tags: list[str]


class desensitization_result(BaseModel):
    res: str


class desensitization_input(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                {
                    "text": "《八佰》（英語：The Eight Hundred）是一部于2020年上映的以中国历史上的战争为题材的电影，由管虎执导，黄志忠、黄骏豪、张俊一、张一山....."
                }
            }
        }


@app.get("/get_filter_conditions_of_law", response_model=conditions_law_result)
def _get_filter_conditions() -> dict:
    """
    Return:
        @result: dict, 返回法条检索的输入条件
    """
    return get_filter_conditions()


@app.post("/search_laws", response_model=search_laws_result)
def search_laws(search_query: Search_laws_input) -> dict:
    """
    获取法律检索的结果

    参数设定；
    Parameters:
        @query: str, 用户输入的查询内容

        @filter_conditions: dict, 用户输入的条件
            @timeliness: str, 用户输入的时效性
            @types_of_law：str, 用户输入的法律种类
            @scope_of_use：str, 用户输入的使用范围
        @page_number: int, 第几页

        @page_size: int, 页大小
    Return:
        @result: list, 搜索返回的结果集合
        @total_amount: int, 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度
    """
    try:
        result = _get_law_result(
            search_query.query,
            search_query.filter_conditions,
            search_query.page_number,
            search_query.page_size,
        )
        return result
    except Exception as e:
        return response_failed_result("error:" + repr(e))


@app.get("/get_law_by_law_id")
def get_law_by_law_id(law_id: Law_id):
    # TODO 待更新
    return {"result": ""}


@app.get("/get_filter_conditions_of_case", response_model=conditions_case_result)
def _get_filter_conditions() -> dict:
    """
    Return:
        @result: dict, 返回案例检索的输入条件
    """
    return get_filter_conditions_of_case()


@app.post("/search_cases", response_model=search_cases_result)
def search_laws(search_query: Search_cases_input) -> dict:
    """
    获取法律检索的结果

    参数设定；
    Parameters:
        @query: str, 用户输入的查询内容

        @filter_conditions: dict, 用户输入的条件
            @type_of_case: list, 用户输入的案件类型
            @court_level: list, 用户输入的法院层级
            @type_of_document: list, 用户输入的文书类型
            @region: list, 用户输入的地域

        @page_number: int, 第几页

        @page_size: int, 页大小
    Return:
        @result: list, 搜索的返回结果
        @total_amount: int 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度
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


@app.post("/top_k_similar_narrative", response_model=similar_narrative_result)
def get_similar_case(search_query: Similar_case_input) -> dict:
    """
    Parameters:
        @fact: str, 用户输入的事实描述
        @problem: str, 用户输入的纠纷类型
        @claim_list: list, 诉求类型
    Return:
        @dids: list, 裁判文书的uq_id
        @sims: list, 相似类案的相似率
        @reasonNames: list, 纠纷类型
        @tags: list, 关键词
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

            return {
                "dids": doc_id_list,
                "sims": sim_list,
                # "winLos": win_los_list,
                "reasonNames": reason_name_list,
                # "appealNames": appeal_name_list,
                "tags": tags_list,
                # "pubDates": pubDate_list,
                # "keywords": keywords,
                # "error_msg": "",
                # "status": 0,
            }

        else:
            return {"error_msg": "data is None", "status": 1}
    except Exception as e:
        return {"error_msg:": repr(e), "status": 1}


@app.post("/desensitization", response_model=desensitization_result)
def get_text_desen(dese_in: desensitization_input) -> str:
    """
    Parameters:
        @text: str, 待脱敏文本
    Return:
        @res: str, 返回脱敏后的结果
    """
    return desensitization(dese_in.text)


if __name__ == "__main__":
    # 日志设置
    uvicorn.run(
        "OnlineServer.ProfessionalSearch.server:app",
        host="0.0.0.0",
        port=8132,
        reload=False,
        workers=1,
    )
