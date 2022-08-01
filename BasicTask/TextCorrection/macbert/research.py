#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/29 15:20
# @Author  : Adolf
# @Site    : 
# @File    : research.py
# @Software: PyCharm
import os
import torch
import operator
from transformers import BertTokenizer, BertForMaskedLM
# import pycorrector

from loguru import logger

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


tokenizer = BertTokenizer.from_pretrained("model/language_model/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("model/language_model/macbert4csc-base-chinese")
model.to(device)

texts = '虽然我从来没有这样凝练的概括过自己的认知观，但是我学到这里的时候也会有一种很强烈感同身受。在我的个人理解中，从最初级的环境到最高级的愿景，其实是完成了一种蜕变，不是说拥有这愿景思维的人就不受环境制约，而是他们在认清客观现实以后的基础上坚持自己的初心，从而去完成自己的所想做的事。'
with torch.no_grad():
    outputs = model(**tokenizer(texts, padding=True, return_tensors='pt').to(device))

res = outputs.logits.squeeze()
# print(res.size())
# exit()
text = tokenizer.decode(torch.argmax(res, dim=-1), skip_special_tokens=True).replace(' ', '')
# print(res)

print(text)
corrected_text, sub_details = get_errors(text, texts)
print(sub_details)