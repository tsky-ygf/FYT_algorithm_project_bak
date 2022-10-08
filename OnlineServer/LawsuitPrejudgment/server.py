#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : server.py
# @Software: PyCharm
import typing

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from LawsuitPrejudgment.service_use import criminal_app_service, administrative_app_service

app = FastAPI()


@app.get("/get_administrative_type")
def _get_administrative_type():
    return administrative_app_service.get_supported_administrative_types()


@app.get('/get_administrative_problem_and_situation_by_type_id')
def _get_administrative_problem_and_situation_by_type_id(type_id: str):
    return administrative_app_service.get_administrative_problem_and_situation_by_type_id(type_id)


class AdministrativeInput(BaseModel):
    type_id: str = "tax"
    situation: str = "逃避税务机关检查"


@app.post('/get_administrative_result')
def _get_administrative_result(param: AdministrativeInput):
    return administrative_app_service.get_administrative_result(param.type_id, param.situation)


class CriminalInput(BaseModel):
    fact: str
    question_answers: typing.Dict
    factor_sentence_list: typing.List


@app.post("/get_criminal_result")
def _get_criminal_result(param: CriminalInput):
    return criminal_app_service.get_criminal_result(param.fact, param.question_answers, param.factor_sentence_list)


if __name__ == '__main__':
    uvicorn.run('OnlineServer.LawsuitPrejudgment.server:app', host="0.0.0.0", port=8105, reload=True, workers=1)
