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

# from typing import Optional

app = FastAPI()


class QueryInput(BaseModel):
    question: str = Field(default="公司交不起税怎么办", description="用户输入的问题")
    source_end: str = Field(default="通用", description="用户使用的客户端")
    query_type: str = Field(default="专题", description="选择用户输入的问题类型")
    query_sub_type: str = Field(default="通用", description="选择用户输入的问题子类型")


@app.post("/get_query_answer_with_source")
def _get_query_answer_with_source(query_input: QueryInput):
    """
    获取智能咨询的结果

    输入参数:

    @question: str , 用户输入的问题

    @source_end: str, 选择用户的类型，可选项:(来源通用/个人端/企业端/律师端)

    @query_type: str, 选择一级主题分类，可选项:(选在旧版/专题)

    @query_sub_type: str, 选择二级主题分类，可选项:(通用/法院/公安/环保/交通/金融/科技/市场监管/税务/司法/文旅)

    输出参数:

    @answer: str, 智能咨询的答案

    @similarity_question: Optional[List]: 返回的相似问题
    """
    print("input:")
    print("question{}:".format(query_input.question))
    print("source_end{}:".format(query_input.source_end))
    print("query_source{}:".format(query_input.query_type))
    print("query_sub_source{}:".format(query_input.query_sub_type))

    if query_input.query_type == "旧版":
        return consultation_service.get_query_answer(query_input.question)

    else:
        if query_input.query_sub_type == "通用":
            return consultation_service.get_query_answer_with_source(question=query_input.question,
                                                                     source=query_input.query_sub_type)

        return consultation_service.get_query_answer_with_source(question=query_input.question,
                                                                 source=query_input.query_type,
                                                                 sub_source=query_input.query_sub_type)


if __name__ == '__main__':
    # 真实使用8130端口。测试使用8131端口
    uvicorn.run('OnlineServer.IntelligentConsultation.server:app', host="0.0.0.0", port=8130, reload=False, workers=2)
    # uvicorn.run('OnlineServer.IntelligentConsultation.server:app', host="0.0.0.0", port=8131, reload=True, workers=4)
