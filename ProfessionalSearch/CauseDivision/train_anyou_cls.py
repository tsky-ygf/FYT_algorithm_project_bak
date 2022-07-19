#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 14:17
# @Author  : Adolf
# @Site    : 
# @File    : train_anyou_cls.py
# @Software: PyCharm
import torch
from Tools.train_tool import BaseTrainTool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ProfessionalSearch.CauseDivision.anyou_dataset import LawsAnyouClsDataset


class TrainLawsCls(BaseTrainTool):
    def __init__(self, config_path):
        super(TrainLawsCls, self).__init__(config_path=config_path)
        self.criterion = torch.nn.CrossEntropyLoss()

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["pre_train_tokenizer"])
        model = AutoModelForSequenceClassification.from_pretrained(self.config["pre_train_model"],
                                                                   num_labels=self.config["num_labels"])
        self.logger.debug(model)
        # for param in model.longformer.parameters():
        #     param.requires_grad = True
        return tokenizer, model

    def init_dataset(self):
        train_dataset = LawsAnyouClsDataset(self.tokenizer,
                                        self.config["train_data_path"],)
        valid_dataset = LawsAnyouClsDataset(self.tokenizer,
                                        self.config["dev_data_path"],)
        return train_dataset, valid_dataset

    def cal_loss(self, batch):
        outputs = self.model(**batch)
        # loss = self.criterion(torch.sigmoid(pred.logits), labels)
        loss = outputs.loss
        return loss


if __name__ == '__main__':
    TrainLawsCls(config_path="ProfessionalSearch/Config_bak/anyou_cls.yaml").train_main()
