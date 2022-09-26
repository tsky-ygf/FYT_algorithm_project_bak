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

from DocumentReview.ContractReview.contract_for_server import *

app = FastAPI()

init_model()


@app.get('/get_contract_type')
def _get_support_contract_types():
    return {"result": get_support_contract_types()}


@app.get('/get_user_standpoint')
def _get_user_standpoint():
    return {"result": get_user_standpoint()}


class Item(BaseModel):
    text: str = "何谓耐受性？"


@app.post("/administrative_consult")
async def get_other_consult(item: Item):
    print("input item: ", item)
    answer = faq_predict(item.text)
    return {"result": answer}


if __name__ == "__main__":
    # 日志设置
    uvicorn.run('ProfessionalSearch.Consult.FAQ_server:app', host="0.0.0.0", port=8110, reload=False, workers=4)
