#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 15:36
# @Author  : Adolf
# @Site    : 
# @File    : use_simbert.py
# @Software: PyCharm
from transformers import BertTokenizer, BertModel
from pprint import pprint, pformat

model_path = 'model/language_model/simbert-chinese-base'
tokenizers = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)
model.eval()
model.cuda()

input0 = ['我是一个中国人', '我是一个美国人', '我不是一个英国人']
input_ids = tokenizers(input0, return_tensors="pt", padding=True, truncation=True)
# print(input_ids)
input_ids = {key:value.cuda() for key,value in input_ids.items()}
output = model(**input_ids)
print(output[1].shape)
print(output.pooler_output.shape)
