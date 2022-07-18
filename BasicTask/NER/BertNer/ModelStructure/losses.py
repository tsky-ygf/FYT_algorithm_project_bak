#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/28 19:02
# @Author  : Adolf
# @Site    : 
# @File    : losses.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-class Focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        :param input: [N, C]
        :param target: [N, ]
        :return:
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, ignore_index=self.ignore_index, reduction='sum')
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self,output,target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)

        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)