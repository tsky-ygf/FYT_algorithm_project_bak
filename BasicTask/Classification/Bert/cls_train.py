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
    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["pre_train_tokenizer"])
        # model = MultiLabelClsModel(self.config)
        model_config = AutoConfig.from_pretrained(self.config["pre_train_model"], num_labels=self.config["num_labels"])
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["pre_train_model"],
            config=model_config,
            from_tf=False,
        )

        return tokenizer, model

    def create_examples(self, data_path, mode):
        pass


if __name__ == '__main__':
    TrainClassification(config_path="BasicTask/Classification/Bert/base_cls_train.yaml").run()
