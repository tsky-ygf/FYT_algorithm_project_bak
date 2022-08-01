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

texts = '整体上培训对我的收获还是非常之大，也有很多很多的心得感悟，以上只是一部分自己的随感而发，也因文笔功力也实属有限，再写怕成了懒婆娘的脚。'
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