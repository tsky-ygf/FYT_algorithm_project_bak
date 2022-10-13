#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/30 11:16
# @Author  : Adolf
# @Site    : 
# @File    : consultation_service.py
# @Software: PyCharm
import requests
import pandas as pd
from IntelligentConsultation.src.faq_pipeline.infer import FAQPredict

df_individual = pd.read_csv("IntelligentConsultation/config/个人端热门标签问答.csv", encoding="utf-8")
df_company = pd.read_csv("IntelligentConsultation/config/企业端热门标签问答.csv", encoding="utf-8")


def _get_initial_answer(question: str):
    for index, row in df_individual.iterrows():
        if str(row["问题"]).strip() == str(question).strip():
            return str(row["回答"])
    for index, row in df_company.iterrows():
        if str(row["问题"]).strip() == str(question).strip():
            return str(row["回答"])
    return None


def _post_process(answer: str):
    return str(answer).replace("\r\n", "\n")


def get_query_answer(question: str):
    initial_answer = _get_initial_answer(question)
    if initial_answer:
        return {
            "answer": _post_process(initial_answer)
        }

    url = "http://172.19.82.198:5050/get_query_answer"
    resp_json = requests.post(url, json={"question": question}).json()
    return {
        "answer": _post_process(resp_json.get("answer"))
    }


faq_predict = FAQPredict(level="INFO",
                         console=True,
                         logger_file="log/intelligent_consultation/model.log",
                         index_name="topic_qa",
                         model_name="model/similarity_model/simcse-model-topic_qa")


def get_query_answer_with_source(question: str, query_type: str):
    answer, similarity_question = faq_predict(question, query_type)
    return {"answer": answer, "similarity_question": similarity_question}
