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
from IntelligentConsultation.core import intelligent_consultation_service as service

app = FastAPI()


class QueryInput(BaseModel):
    question: str = "你好吗？"
    source_end: str = None


@app.post("/get_query_answer")
def _get_query_answer(query_input: QueryInput):
    return service.get_query_answer(query_input.question)


if __name__ == '__main__':
    uvicorn.run('OnlineServer.IntelligentConsultation.server:app', host="0.0.0.0", port=8130, reload=False, workers=1)
