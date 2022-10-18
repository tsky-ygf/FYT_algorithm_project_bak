#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/14 14:12
# @Author  : Adolf
# @Site    : 
# @File    : insert_data_to_es.py
# @Software: PyCharm
from IntelligentConsultation.src.faq_pipeline.main import insert_data

# 专题数据
index_name = "topic_qa"
data_path = "data/fyt_train_use_data/QA/pro_qa.csv"
model_path = "model/similarity_model/simcse-model-topic-qa"

insert_data(data_path=data_path, index_name=index_name, model_name=model_path)
