#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 16:32
# @Author  : Adolf
# @Site    : 
# @File    : use_law_language_model.py
# @Software: PyCharm
from transformers import AutoModel, AutoTokenizer

tokenizer_path = "model/language_model/chinese-roberta-wwm-ext"
model_path = "model/language_model/lawformer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(model_path)

inputs = tokenizer("任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", return_tensors="pt")
outputs = model(**inputs)

print(outputs)
