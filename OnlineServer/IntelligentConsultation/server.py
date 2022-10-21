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
    query_type: str = Field(default="专题", description="选择一级主题分类，可选项:(旧版/专题)")
    query_sub_type: str = Field(default="通用",
                                description="选择二级主题分类，可选项:(通用/法院/公安/环保/交通/金融/科技/市场监管/税务/司法/文旅)")

    class Config:
        schema_extra = {
            "example": {
                "question": "公司交不起税怎么办",
                "query_type": "专题",
                "query_sub_type": "通用",
            }
        }


class QueryResult(BaseModel):
    answer: str
    similarity_question: list[dict]


@app.post("/get_query_answer_with_source", response_model=QueryResult)
def _get_query_answer_with_source(query_input: QueryInput):
    """
    获取智能咨询的结果和相似问答

    请求参数:

    | Param           | Type | Description                                       |
    |-----------------|------|---------------------------------------------------|
    | question        | str  | 用户输入的问题                                           |
    | query_type      | str  | 选择一级主题分类，可选项:(旧版/专题)                              |
    | query_sub_type  | str  | 选择二级主题分类，可选项:(通用/法院/公安/环保/交通/金融/科技/市场监管/税务/司法/文旅) |

    响应参数:

    | Param               | Type           | Description |
    |---------------------|----------------|-------------|
    | answer              | str            | 智能咨询的答案     |
    | similarity_question | Optional[List] | 返回的相似问题     |

    """
    return consultation_service.get_answer_for_question(question=query_input.question,
                                                        query_type=query_input.query_type,
                                                        query_sub_type=query_input.query_sub_type)


if __name__ == '__main__':
    # 真实使用8134端口
    uvicorn.run('OnlineServer.IntelligentConsultation.server:app', host="0.0.0.0", port=8134, reload=False, workers=2)
