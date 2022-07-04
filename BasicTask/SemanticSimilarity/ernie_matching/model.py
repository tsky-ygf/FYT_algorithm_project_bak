#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 14:16
# @Author  : Adolf
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseMatching(nn.Module):
    def __init__(self, pretrained_model, dropout=None, margin=0.1):
        super(PairwiseMatching, self).__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin

        self.similarity = nn.Linear(self.ptm.config.hidden_size, 1)

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        sim = self.similarity(cls_embedding)
        return sim

    def predict(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)
        sim_score = self.similarity(cls_embedding)
        sim_score = F.sigmoid(sim_score)

        return sim_score

    def forward(self,
                pos_input_ids,
                neg_input_ids,
                pos_token_type_ids=None,
                neg_token_type_ids=None,
                pos_position_ids=None,
                neg_position_ids=None,
                pos_attention_mask=None,
                neg_attention_mask=None):
        _, pos_cls_embedding = self.ptm(pos_input_ids, pos_token_type_ids,
                                        pos_position_ids, pos_attention_mask)

        _, neg_cls_embedding = self.ptm(neg_input_ids, neg_token_type_ids,
                                        neg_position_ids, neg_attention_mask)

        pos_embedding = self.dropout(pos_cls_embedding)
        neg_embedding = self.dropout(neg_cls_embedding)

        pos_sim = self.similarity(pos_embedding)
        neg_sim = self.similarity(neg_embedding)

        pos_sim = F.sigmoid(pos_sim)
        neg_sim = F.sigmoid(neg_sim)

        labels = torch.full(size=[pos_cls_embedding.shape[0]], fill_value=1.0)

        loss = F.margin_ranking_loss(pos_sim,
                                     neg_sim,
                                     labels,
                                     margin=self.margin)

        return loss
