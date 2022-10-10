#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/6 15:53
# @Author  : Adolf
# @Site    : 
# @File    : server_template.py
# @Software: PyCharm
from fastapi import FastAPI
from pydantic import BaseModel

from LawsuitPrejudgment.src.criminal.extraction.feature_extraction import (
    init_extract,
    post_process_uie_results,
)

criminal_list = ["theft", "provide_drug"]
predictor_dict = {}
for criminal_type in criminal_list:
    predictor_dict[criminal_type] = init_extract(criminal_type=criminal_type)

app = FastAPI()


class Data(BaseModel):
    criminal_type: str
    fact: str


@app.post("/information_result")
async def get_information_result(data: Data):
    # result = predictor_dict[_criminal_type].predict([_fact])
    result = post_process_uie_results(
        predictor_dict[data.criminal_type], data.criminal_type, data.fact
    )
    return {"result": result}

# uvicorn LawsuitPrejudgment.Criminal.extraction.server_template:app --port 7777 --host 0.0.0.0
