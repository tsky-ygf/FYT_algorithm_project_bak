#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/21 14:07
# @Author  : Adolf
# @Site    : 
# @File    : extractive_qa_train.py
# @Software: PyCharm
# import torch
from Tools.train_tool import BaseTrainTool
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering


# from Tools.data_pipeline import InputExample


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def create_examples(data_path=""):
    examples = []
    # examples.append(InputExample(guid=guid, text=text, label=[y_onehot]))
    return examples


class TrainExtractQA(BaseTrainTool):
    def __init__(self, config_path, data_func):
        super(TrainExtractQA, self).__init__(config_path=config_path, data_func=data_func)
        exit()

    # self.create_examples = self

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name)
        # model = MultiLabelClsModel(self.config)
        model_config = AutoConfig.from_pretrained(self.model_args.config_name)
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_args.model_name_or_path,
            config=model_config,
        )
        # self.logger.debug(model)

        # for param in model.longformer.parameters():
        #     param.requires_grad = False

        return tokenizer, model

    def cal_loss(self, batch):
        self.logger.debug(batch)
    # self.logger.debug(batch)
    # labels = batch['labels']
    # input_data = {'input_ids': batch['input_ids'],
    #               'atention_mask': batch['attention_mask'],
    #               'token_type_ids': batch['token_type_ids']}
    # pred = self.model(batch['input_ids'].squeeze())
    # loss = self.criterion(torch.sigmoid(pred.logits), labels.squeeze())
    # self.logger.debug(loss)
    # return loss


if __name__ == '__main__':
    TrainExtractQA(config_path="BasicTask/QA/Extractive-QA/base_qa.yaml",
                   data_func=create_examples).run()
