#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/9 13:48
# @Author  : Adolf
# @Site    : 
# @File    : st_losses.py
# @Software: PyCharm
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer, util

import torch.nn.functional as F


class RDropLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, reduction: str = 'none'):
        """
        R-Drop Loss implementation
        For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        Original implementation please refer to this code: https://github.com/dropreg/R-Drop

        Args:
            reduction(str, optional):
                Indicate how to average the loss, the candicates are ``'none'``,``'batchmean'``,``'mean'``,``'sum'``.
                If `reduction` is ``'mean'``, the reduced mean loss is returned;
                If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
                If `reduction` is ``'sum'``, the reduced sum loss is returned;
                If `reduction` is ``'none'``, no reduction will be applied.
                Defaults to ``'none'``.
        """
        super(RDropLoss, self).__init__()
        self.model = model
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                "'reduction' in 'RDropLoss' should be 'sum', 'mean' 'batchmean', or 'none', "
                "but received {}.".format(reduction))
        self.reduction = reduction
        self.pad_mask = None

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        p = reps[0]
        q = reps[1]

        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction=self.reduction)
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction=self.reduction)

        # pad_mask is for seq-level tasks
        if self.pad_mask is not None:
            p_loss.masked_fill_(self.pad_mask, 0.)
            q_loss.masked_fill_(self.pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss

    def get_config_dict(self):
        return {'reduction': self.reduction}


class MNRRDropLoss(nn.Module):

    def __init__(self,
                 model: SentenceTransformer,
                 scale: float = 20.0,
                 similarity_fct=util.cos_sim,
                 reduction: str = 'none',
                 kl_weight: float = 0.1):
        super().__init__()
        self.model = model

        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.pad_mask = None

        self.reduction = reduction
        self.kl_weight = kl_weight

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        multi_neg_loss = self.cal_multi_neg_loss(reps[0], torch.cat(reps[1:]))
        kl_loss = self.cal_kl_loss(reps[0], torch.cat(reps[1:]))

        # return multi_neg_loss
        return multi_neg_loss + kl_loss * self.kl_weight

    def cal_multi_neg_loss(self, embeddings_a, embeddings_b):
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]
        multi_neg_loss = self.cross_entropy_loss(scores, labels)
        return multi_neg_loss

    def cal_kl_loss(self, p, q):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1),
                          reduction=self.reduction)
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1),
                          reduction=self.reduction)

        # pad_mask is for seq-level tasks
        if self.pad_mask is not None:
            p_loss.masked_fill_(self.pad_mask, 0.)
            q_loss.masked_fill_(self.pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        kl_loss = (p_loss + q_loss) / 2

        return kl_loss

    def get_config_dict(self):
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__, "kl_weight": self.kl_weight}
