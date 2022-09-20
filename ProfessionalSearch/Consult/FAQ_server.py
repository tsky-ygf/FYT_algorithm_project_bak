#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 10:37
# @Author  : Adolf
# @Site    : 
# @File    : FAQ_server.py
# @Software: PyCharm
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ProfessionalSearch.Consult.FAQ_predict import FAQPredict

app = FastAPI()

faq_predict = FAQPredict()


class Item(BaseModel):
    text: str = "何谓耐受性？"


@app.post("/administrative_consult")
async def get_other_consult(item: Item):
    print("input item: ", item)
    answer = faq_predict(item.text)
    return {"result": answer}


if __name__ == "__main__":
    # 日志设置
    uvicorn.run('ProfessionalSearch.Consult.FAQ_server:app', host="0.0.0.0", port=8127, reload=False)
