#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/09 15:52
# @Author  : Czq
# @File    : model_NER.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class PointerNERBERTInFramework(nn.Module):
    def __init__(self, args):
        super(PointerNERBERTInFramework, self).__init__()

        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.num_labels = args.num_labels
        self.linear_hidden = nn.Linear(args.bert_emb_size, args.hidden_size)
        self.linear_start = nn.Linear(args.hidden_size, self.num_labels)
        self.linear_end = nn.Linear(args.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        bert_emb = self.bert(**inputs)
        bert_out, bert_pool = bert_emb[0], bert_emb[1]

        hidden = self.linear_hidden(bert_out)
        start_logits = self.linear_start(self.gelu(hidden))
        end_logits = self.linear_end(self.gelu(hidden))
        # delete cls and sep
        start_logits = start_logits[:, 1:-1]
        end_logits = end_logits[:, 1:-1]
        start_prob = self.sigmoid(start_logits)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob

