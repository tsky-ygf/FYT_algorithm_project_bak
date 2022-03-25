#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:59
# @Author  : Adolf
# @Site    : 
# @File    : train_laws_cls.py
# @Software: PyCharm
from RelevantLaws.Tools.train_tool import BaseTrainTool
from transformers import AutoModel, AutoTokenizer


class TrainLawsCls(BaseTrainTool):
    def __init__(self, config_path):
        super(TrainLawsCls, self).__init__(config_path=config_path)

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_path"])
        model = AutoModel.from_pretrained(self.config["model_path"])
        return tokenizer, model

    def init_dataset(self, *args, **kwargs):
        raise NotImplemented

    def cal_loss(self, *args, **kwargs):
        pass

    def eval_epoch(self):
        pass
