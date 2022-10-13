#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/13 10:23
# @Author  : Adolf
# @Site    : 
# @File    : server.py
# @Software: PyCharm
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from IntelligentConsultation.service_use import consultation_service

app = FastAPI()


class QueryInput(BaseModel):
    question: str = Field(default="公司交不起税怎么办", description="用户输入的问题")
    source_end: str = Field(default=None, description="用户使用的客户端")
    query_type: str = Field(default="专题", description="选择用户输入的问题类型")


@app.post("/get_query_answer")
def _get_query_answer(query_input: QueryInput):
    """
    获取线上智能咨询的结果(废弃）
    """
    print("input:")
    print("question{}:".format(query_input.question))
    print("source_end{}:".format(query_input.source_end))
    print("query_type{}:".format(query_input.query_type))

    return consultation_service.get_query_answer(query_input.question)


@app.post("/get_query_answer_with_source")
def _get_query_answer_with_source(query_input: QueryInput):
    """
    获取智能咨询的结果

    参数设定；

    @question: 用户的问题输入

    @source_end: 个人端/企业端/律师端

    @query_type: 旧版/专题/市场监管/金融/税务/司法
    """
    print("input:")
    print("question{}:".format(query_input.question))
    print("source_end{}:".format(query_input.source_end))
    print("query_type{}:".format(query_input.query_type))

    if query_input.query_type == "旧版":
        return consultation_service.get_query_answer(query_input.question)

    else:
        return consultation_service.get_query_answer_with_source(query_input.question,
                                                                 query_type=query_input.query_type)


if __name__ == '__main__':
    # 真实使用8130端口。测试使用8131端口
    uvicorn.run('OnlineServer.IntelligentConsultation.server:app', host="0.0.0.0", port=8130, reload=False, workers=2)
    # uvicorn.run('OnlineServer.IntelligentConsultation.server:app', host="0.0.0.0", port=8131, reload=True, workers=4)
