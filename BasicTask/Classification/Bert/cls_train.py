# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:59
# @Author  : Adolf
# @Site    : 
# @File    : cls_train.py
# @Software: PyCharm
# import os
import torch
from Tools.train_tool import BaseTrainTool
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TrainClassification(BaseTrainTool):
    def __init__(self, config, create_examples):
        super(TrainClassification, self).__init__(config=config, create_examples=create_examples)
        self.criterion = torch.nn.BCELoss()

    # self.create_examples = self

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["pre_train_tokenizer"])
        # model = MultiLabelClsModel(self.config)
        model_config = AutoConfig.from_pretrained(self.config["pre_train_model"], num_labels=self.config["num_labels"])
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["pre_train_model"],
            config=model_config,
            from_tf=False,
        )
        # self.logger.debug(model)

        # for param in model.longformer.parameters():
        #     param.requires_grad = False

        return tokenizer, model

    def cal_loss(self, batch):
        self.logger.debug(batch)
        labels = batch['labels']
        # input_data = {'input_ids': batch['input_ids'],
        #               'atention_mask': batch['attention_mask'],
        #               'token_type_ids': batch['token_type_ids']}
        pred = self.model(batch['input_ids'].squeeze())
        loss = self.criterion(torch.sigmoid(pred.logits), labels.squeeze())
        self.logger.debug(loss)
        return loss

# if __name__ == '__main__':
#     TrainClassification(config_path="BasicTask/Classification/Bert/base_cls_train.yaml").train_main()
#     # TrainLawsCls(config_path="RelevantLaws/Config/base_cls_train.yaml").eval_epoch()
