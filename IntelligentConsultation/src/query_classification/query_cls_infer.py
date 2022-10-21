#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/21 09:27
# @Author  : Adolf
# @Site    : 
# @File    : query_cls_infer.py
# @Software: PyCharm
from transformers import BertTokenizer, BertForSequenceClassification


def init_torch_model(model_path):
    tokenizers = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.cpu()

    return tokenizers, model


def init_config():
    label_path = "IntelligentConsultation/config/query_cls_label.txt"
    label_map = {}
    with open(label_path, "r") as f:
        for line in f.readlines():
            line_list = line.strip().split(",")
            label_map[int(line_list[1])] = line_list[0]

    return label_map


def get_torch_model_result(tokenizers, model, input_text):
    input_ids = tokenizers(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # input_ids = {key: value.cuda() for key, value in input_ids.items()}
    outputs = model(**input_ids)
    predictions = outputs.logits.argmax(dim=-1)
    return predictions


if __name__ == '__main__':
    _label_map = init_config()
    # print(label_map)
    _tokenizers, _model = init_torch_model("model/similarity_model/query_cls/final")
    res = get_torch_model_result(_tokenizers, _model, "我被邻居家的狗咬了怎么办？")
    print(_label_map[res[0].item()])
