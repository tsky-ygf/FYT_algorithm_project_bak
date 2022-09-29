#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : server.py
# @Software: PyCharm
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from IntelligentConsultation.service_use import intelligent_consultation_service

app = FastAPI()


class QueryInput(BaseModel):
    question: str = "公司交不起税了怎么办？"
    source_end: str = None


@app.post("/get_query_answer")
def _get_query_answer(query_input: QueryInput):
    """
    获取线上咨询的答案
    :param query_input:
    :return:
    """
    print("input:")
    print("question{}:".format(query_input.question))
    print("source_end{}:".format(query_input.source_end))

    return intelligent_consultation_service.get_query_answer(query_input.question)


@app.post("/get_query_answer_with_source")
def _get_query_answer_with_source(query_input: QueryInput):
    """
    获取专题咨询的内容
    :param query_input:
    :return:
    """
    print("input:")
    print("question{}:".format(query_input.question))
    print("source_end{}:".format(query_input.source_end))

    return intelligent_consultation_service.get_query_answer_with_source(query_input.question)


if __name__ == '__main__':
    uvicorn.run('OnlineServer.IntelligentConsultation.server:app', host="0.0.0.0", port=8130, reload=False, workers=1)
