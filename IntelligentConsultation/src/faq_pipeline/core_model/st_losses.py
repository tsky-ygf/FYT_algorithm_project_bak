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

    def __init__(self, model: SentenceTransformer, reduction: str = 'none', scale: float = 20.0,
                 similarity_fct=util.cos_sim):
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

        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, p, q, pad_mask=None):
        """
        Args:
            p(Tensor): the first forward logits of training examples.
            q(Tensor): the second forward logits of training examples.
            pad_mask(Tensor, optional): The Tensor containing the binary mask to index with, it's data type is bool.

        Returns:
            Tensor: Returns tensor `loss`, the rdrop loss of p and q.
        """
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        rep_anchor, rep_pos, rep_neg = reps
        # distance_pos = self.distance_metric(rep_anchor, rep_pos)
        # distance_neg = self.distance_metric(rep_anchor, rep_neg)

        # losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        # return losses.mean()

        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction=self.reduction)
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction=self.reduction)

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss

    def get_config_dict(self):
        return {'reduction': self.reduction}
