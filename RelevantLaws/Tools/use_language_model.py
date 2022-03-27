#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 00:07
# @Author  : Adolf
# @Site    : 
# @File    : use_language_model.py
# @Software: PyCharm
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model/language_model/chinese-roberta-wwm-ext/")
model = AutoModel.from_pretrained("model/language_model/lawformer")
inputs = tokenizer("任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", return_tensors="pt")
outputs = model(**inputs)
print(outputs['last_hidden_state'])