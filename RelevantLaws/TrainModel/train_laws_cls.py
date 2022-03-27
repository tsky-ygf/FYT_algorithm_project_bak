#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:59
# @Author  : Adolf
# @Site    : 
# @File    : train_laws_cls.py
# @Software: PyCharm
import torch

from RelevantLaws.Tools.train_tool import BaseTrainTool
from RelevantLaws.DataProcess.laws_model_dataset import LawsThuNLPDataset
from RelevantLaws.ModelFile.multi_label_model import MultiLabelClsModel
from transformers import AutoTokenizer


class TrainLawsCls(BaseTrainTool):
    def __init__(self, config_path):
        super(TrainLawsCls, self).__init__(config_path=config_path)
        self.criterion = torch.nn.BCELoss()

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["pre_train_tokenizer"])
        model = MultiLabelClsModel(self.config)
        return tokenizer, model

    def init_dataset(self):
        train_dataset = LawsThuNLPDataset(self.tokenizer,
                                          self.config["train_data_path"],
                                          self.config["label_mapping_path"]
                                          )
        valid_dataset = LawsThuNLPDataset(self.tokenizer,
                                          self.config["dev_data_path"],
                                          self.config["label_mapping_path"]
                                          )
        return train_dataset, valid_dataset

    def data_collator(self, batch):
        #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
        self.logger.debug(batch)
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        labels_list = []
        for one_data in batch:
            input_ids_list.append(one_data['input_ids'])
            token_type_ids_list.append(one_data['token_type_ids'])
            attention_mask_list.append(one_data['attention_mask'])
            labels_list.append(one_data['labels'])

        input_ids = torch.stack(input_ids_list, dim=0)
        token_type_ids = torch.stack(token_type_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        input_ids = torch.squeeze(input_ids)
        token_type_ids = torch.squeeze(token_type_ids)
        attention_mask = torch.squeeze(attention_mask)

        labels = torch.stack(labels_list, dim=0)

        input_data = {"input_ids": input_ids,
                      "token_type_ids": token_type_ids,
                      "attention_mask": attention_mask, }

        return input_data, labels

    def cal_loss(self, batch):
        # self.logger.info(batch)
        input_data, labels = batch
        pred = self.model(input_data)
        loss = self.criterion(torch.sigmoid(pred), labels)
        self.logger.debug(loss)

        return loss

    def eval_epoch(self):
        pass


if __name__ == '__main__':
    TrainLawsCls(config_path="RelevantLaws/Config/base_laws_cls_model.yaml").train_main()
