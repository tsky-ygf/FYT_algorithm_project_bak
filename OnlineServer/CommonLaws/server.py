#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    :
# @File    : server.py
# @Software: PyCharm
# import _io
# import time

import pymysql
import uvicorn
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from CommonLaws.service_use import common_laws_service
from pydantic import BaseModel, Field

app = FastAPI()

# @app.exception_handler(RequestValidationError)
# async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
#     print(f"参数不对{request.method} {request.url}")
#     return JSONResponse({"code": "400", "error_msg": exc.errors(),"status": 1})

class CategoryInput(BaseModel):
    category: str = Field(default="", description="专栏名称:可选项:(税法专栏/司法专栏/金融专栏/市场监督/法院专栏/公安专栏/文旅专栏/环保专栏/交通专栏/科技专栏)")
    class Config:
        schema_extra = {
            "example": {
                "category": "税法专栏",
            }
        }
class CategoryResult(BaseModel):
    data_list: list[dict]

@app.post("/exampleData", response_model=CategoryResult)
async def get_example_model_data(category: CategoryInput):
    tabName = common_laws_service.get_table(category.category)
    preview_data_list = common_laws_service.get_preview_data(tableName=tabName)
    return {"data_list":preview_data_list}

class NewsInput(BaseModel):
    table_name: str = Field(default="", description="数据库表名称")
    uq_id: str = Field(default="", description="数据表主键")

    class Config:
        schema_extra = {
            "example": {
                "table_name": "swj_hot_news",
                "query_type": "e9b672a97cf1d68e57d1cbcd38f6237f",
            }
        }

class NewsResult(BaseModel):
    data_list: list[dict]

@app.post("/getNews", response_model=NewsResult)
async def get_news(info_input: NewsInput):
    data_list = common_laws_service.get_news(uq_id=info_input.uq_id,
                                             tableName=info_input.table_name)
    return {"data_list":data_list}

if __name__ == "__main__":
    # 日志设置
    uvicorn.run('OnlineServer.CommonLaws.server:app', host="0.0.0.0", port=8149, reload=False, workers=1)
