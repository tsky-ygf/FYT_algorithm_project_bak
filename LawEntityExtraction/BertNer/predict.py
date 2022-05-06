#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/5 12:03
# @Author  : Adolf
# @Site    : 
# @File    : predict.py
# @Software: PyCharm
import torch
from transformers import BertTokenizer
from LawEntityExtraction.BertNer.ModelStructure.bert_ner_model import BertSpanForNer
from Utils.parse_file import parse_config_file


def bert_extract_item(_start_logits, _end_logits):
    S = []
    start_pred = torch.argmax(_start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(_end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S


config = parse_config_file("LawEntityExtraction/BertNer/base_ner_config.yaml")
tokenizer = BertTokenizer.from_pretrained("model/laws_ner_clue/final/")
model = BertSpanForNer(config)
model.load_state_dict(torch.load("model/laws_ner_clue/final/pytorch_model.bin"))
model.eval()
# "text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，",
# "label": {"address": {"台湾": [[15, 16]]}, "name": {"彭小军": [[0, 2]]}}
txt = "姓名王飞性别女民族汉出生1992年8月18日住址辽宁省大连市甘井子区公民身份证号码52231589754511"
inputs = tokenizer(txt, add_special_tokens=True,
                   # pahudding=True,
                   max_length=128,
                   # truncation=False,
                   return_offsets_mapping=False,
                   return_tensors="pt")

print(inputs)
outputs = model.forward(input_ids=inputs['input_ids'],
                token_type_ids=inputs['token_type_ids'],
                attention_mask=inputs['attention_mask'])

start_logits, end_logits = outputs[:2]
R = bert_extract_item(start_logits, end_logits)

label_list = ["O", "address", "book", "company", 'game', 'government', 'movie', 'name', 'organization', 'position',
              'scene']
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

if R:
    label_entities = [[id2label[x[0]], x[1], x[2]] for x in R]
else:
    label_entities = []

words = list(txt)
json_d = {}
json_d['label'] = {}
if len(label_entities) != 0:
    for subject in label_entities:
        tag = subject[0]
        start = subject[1]
        end = subject[2]
        word = "".join(words[start:end + 1])
        if tag in json_d['label']:
            if word in json_d['label'][tag]:
                json_d['label'][tag][word].append([start, end])
            else:
                json_d['label'][tag][word] = [[start, end]]
        else:
            json_d['label'][tag] = {}
            json_d['label'][tag][word] = [[start, end]]

print(json_d)
