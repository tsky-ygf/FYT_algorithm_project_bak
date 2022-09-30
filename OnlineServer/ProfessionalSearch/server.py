#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : server.py
# @Software: PyCharm

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

from ProfessionalSearch.service_use.relevant_laws.relevant_laws_api import get_filter_conditions, \
    _get_law_result
from ProfessionalSearch.service_use.similar_case_retrival.similar_case_retrieval_api import get_filter_conditions_of_case, \
    _get_case_result

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


class Case_id(BaseModel):
    case_id: str


@app.get('/get_filter_conditions_of_law')
def _get_filter_conditions():
    return {"result": get_filter_conditions()}


@app.post('/search_laws')
def search_laws(search_query: Search_laws_input):
    return {"result": _get_law_result(search_query.query, search_query.filter_conditions, search_query.page_number,
                                      search_query.page_size)}


@app.get('/get_law_by_law_id')
def get_law_by_law_id(law_id: Law_id):
    # TODO 待更新
    return {"result": ""}


@app.get('/get_filter_conditions_of_case')
def _get_filter_conditions():
    return {"result": get_filter_conditions_of_case()}


@app.post('/search_cases')
def search_laws(search_query: Search_cases_input):
    return {"result": _get_case_result(search_query.query, search_query.filter_conditions, search_query.page_number,
                                       search_query.page_size)}


@app.get('/get_law_document')
def get_law_document(case_id: Case_id):
    # TODO 待更新
    return {"result": ""}


if __name__ == "__main__":
    # 日志设置
    uvicorn.run('OnlineServer.ProfessionalSearch.server:app', host="0.0.0.0", port=8135, reload=False, workers=1)
