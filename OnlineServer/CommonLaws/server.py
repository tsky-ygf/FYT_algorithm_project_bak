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
    category: str = Field(default="", description="专栏名称->可选项:(税法专栏/司法专栏/金融专栏/市场监督/法院专栏/公安专栏/文旅专栏/环保专栏/交通专栏/科技专栏)")
    class Config:
        schema_extra = {
            "example": {
                "category": "税法专栏",
            }
        }
class CategoryResult(BaseModel):
    data_list: list[dict]

@app.post("/fyt/commonLaws/v1.0.0/get_preview_commonLaws_data", summary="获取专栏预览信息", response_model=CategoryResult)
async def get_example_model_data(category: CategoryInput):
    """
        获取 专栏预览 信息

        请求参数:

        | Param           | Type | Description                                       |
        |-----------------|------|---------------------------------------------------|
        | category        | str  | 专栏名称->可选项:(税法专栏/司法专栏/金融专栏/市场监督/法院专栏/公安专栏/文旅专栏/环保专栏/交通专栏/科技专栏)                                         |

        响应参数:

        | Param       | Type           | Description |
        |-------------|----------------|-------------|
        | uq_id       | str            | 主键 ID      |
        | title       | str            | 标题         |
        | pubDate     | str            | 发布时间      |
        | source      | str            | 来源          |
        | preview     | str            | 预览内容信息   |
        | tableName   | str            | 数据库表名     |

        """
    tabName = common_laws_service.get_table(category.category)
    preview_data_list = common_laws_service.get_preview_data(tableName=tabName)
    return {"data_list":preview_data_list}

class NewsInput(BaseModel):
    table_name: str = Field(default="swj_hot_news", description="数据库表名称")
    uq_id: str = Field(default="e9b672a97cf1d68e57d1cbcd38f6237f", description="数据表主键")

    class Config:
        schema_extra = {
            "example": {
                "table_name": "swj_hot_news",
                "uq_id": "e9b672a97cf1d68e57d1cbcd38f6237f",
            }
        }

class NewsResult(BaseModel):
    data_list: list[dict]

@app.post("/fyt/commonLaws/v1.0.0/get_commonLaws_news_by_id", summary="根据ID查询专栏信息", response_model=NewsResult)
async def get_news(info_input: NewsInput):
    """
        根据 ID 查询专栏详细信息

        请求参数:

        | Param           | Type | Description                                       |
        |-----------------|------|---------------------------------------------------|
        | table_name      | str  | 数据库表 名称                                           |
        | uq_id           | str  | 数据库表 主键ID                            |

        响应参数:

        | Param               | Type           | Description |
        |---------------------|----------------|-------------|
        | url                 | str            | 网站链接     |
        | htmlContent         | str            | 带HTML的原文内容     |
        | title               | str            | 标题     |
        | pubDate             | str            | 发布时间     |
        | source              | str            | 来源     |
        | preview             | str            | 内容预览     |

        """
    data_list = common_laws_service.get_news(uq_id=info_input.uq_id,
                                             tableName=info_input.table_name)
    return {"data_list":data_list}

if __name__ == "__main__":
    # 日志设置
    uvicorn.run('OnlineServer.CommonLaws.server:app', host="0.0.0.0", port=8148, reload=False, workers=1)
