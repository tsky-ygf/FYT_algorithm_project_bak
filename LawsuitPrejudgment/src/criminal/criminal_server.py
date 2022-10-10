#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 11:37
# @Author  : Adolf
# @Site    : 
# @File    : criminal_server.py
# @Software: PyCharm
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from LawsuitPrejudgment.src.criminal.criminal_prejudgment import CriminalPrejudgment

app = FastAPI()

criminal_config = {
    "log_level": "debug",
    # "log_path": "log/lawsuit_prejudgment/criminal_prejudgment.log",
    "prejudgment_type": "criminal",
    "anyou_identify_model_path": "model/gluon_model/accusation",
    "situation_identify_model_path": "http://127.0.0.1:7777/information_result",
}
criminal_pre_judgment = CriminalPrejudgment(**criminal_config)


class Item(BaseModel):
    fact: str = "我偷了同事的3000元"
    question_answers: Dict = {}


@app.post("/criminal_prejudgment")
async def get_criminal_prejudgment(item: Item):
    t0 = time.time()
    print("input item: ", item)
    if item.question_answers == {}:
        criminal_pre_judgment.init_content()
    t1 = time.time()
    print("t1-t0", t1-t0)
    # 改变item.question_answers的格式：将情形作为键值。这样CriminalPrejudgment才能处理。
    question_answers = {}
    for question, answer in item.question_answers.items():
        circumstance, info = criminal_pre_judgment.get_circumstance_of_question(question)
        question_answers[circumstance] = info
        question_answers[circumstance]["usr_answer"] = answer
    item.question_answers = question_answers
    t2 = time.time()
    print("t2-t1", t2-t1)
    criminal_pre_judgment(fact=item.fact, question_answers=item.question_answers)
    t3 = time.time()
    print("t3-t2", t3-t2)
    while "report_result" not in criminal_pre_judgment.content:
        next_question = criminal_pre_judgment.get_next_question()
        t4 = time.time()
        print("t4-t3", t4 - t3)
        return {"next_question": next_question}
    t5 = time.time()
    print("t5-t3", t5 - t3)
    return {"result": criminal_pre_judgment.content["report_result"]}


if __name__ == "__main__":
    # 日志设置
    uvicorn.run('LawsuitPrejudgment.src.criminal.criminal_server:app', host="0.0.0.0", port=5081, reload=True)
