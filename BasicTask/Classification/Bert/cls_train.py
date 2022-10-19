# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 14:59
# @Author  : Adolf
# @Site    : 
# @File    : cls_train.py
# @Software: PyCharm
# import os
import pandas as pd
import torch
from Tools.train_tool import BaseTrainTool
from Tools.data_pipeline import InputExample

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TrainClassification(BaseTrainTool):
    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name)
        model_config = AutoConfig.from_pretrained(self.model_args.config_name, self.model_args.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            config=model_config,
        )
        return tokenizer, model

    def create_examples(self, data_path, mode):
        self.logger.info("Creating {} examples".format(mode))
        self.logger.info("Creating examples from {} ".format(data_path))
        examples = []

        data_df = pd.read_csv(data_path)
        for index, row in data_df.iterrows():
            print(row)
            exit()
            # examples.append(InputExample(guid=guid, texts=text, label=label))
        return examples


if __name__ == '__main__':
    TrainClassification(config_path="BasicTask/Classification/Bert/base_cls_train.yaml").run()
