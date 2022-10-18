#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 10:35
# @Author  : Adolf
# @Site    : 
# @File    : use_ernie-gram.py
# @Software: PyCharm
from transformers import AutoTokenizer, AutoModel

model_path = 'model/language_model/ernie-gram'

tokenizers = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
print(model)

input0 = ['我是一个中国人', '我是一个美国人']
input_ids = tokenizers(input0, return_tensors="pt", padding=True, truncation=True)
print(input_ids)
output = model(**input_ids)
output = output.pooler_output
# print(output)
# print(output.pooler_output.shape)
# output = model.encoder(**input_ids)
print(output.shape)
print(output)
