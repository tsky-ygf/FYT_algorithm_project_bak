#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 14:57
# @Author  : Adolf
# @Site    : 
# @File    : metrics.py
# @Software: PyCharm
from typing import List, Any
from sklearn.metrics import accuracy_score

from Utils.register import Registry

Metrics = Registry('COMMON_METRICS')


@Metrics.register()
class Accuracy:
    predictions: List[Any]
    references: List[Any]

    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.references = []

    @staticmethod
    def compute(predictions, references, normalize=True, sample_weight=None):
        acc = accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
        return acc

    def result(self):
        # class_info = {}
        # for type_, count in origin_counter.items():
        #     origin = count
        #     found = found_counter.get(type_, 0)
        #     right = right_counter.get(type_, 0)
        #     recall, precision, f1 = self.compute(origin, found, right)
        #     class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        # origin = len(self.origins)
        # found = len(self.founds)
        # right = len(self.rights)
        # recall, precision, f1 = self.compute(origin, found, right)
        acc = self.compute(predictions=self.predictions, references=self.references)
        # print("acc:", acc)
        return {'acc': acc}  # , class_info

    # 真实值在前，预测值在后
    def update(self, true_subject, pred_subject):
        self.predictions.extend(pred_subject)
        self.references.extend(true_subject)
