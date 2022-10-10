#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/9 16:07
# @Author  : Adolf
# @Site    : 
# @File    : train.py
# @Software: PyCharm
from Tools.train_tool import BaseTrainTool
from Tools.data_pipeline import InputExample

from transformers import AutoTokenizer
from BasicTask.SentenceEmbedding.SimCSE_ST.model import SimCSE

import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class TrainSimCSE(BaseTrainTool):
    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_args.tokenizer_name)
        # model = MultiLabelClsModel(self.config)
        model = SimCSE(model_name=self.model_args.model_name_or_path, max_seq_length=self.data_train_args.max_length)
        return tokenizer, model

    def create_examples(self, data_path, mode="train"):
        self.logger.info("Creating {} examples".format(mode))
        self.logger.info("Creating examples from {} ".format(data_path))

        train_data = []
        train_df = pd.read_csv(data_path)
        for index, row in train_df.iterrows():
            query = row["question"]
            train_data.append(InputExample(texts=[query, query]))

        return train_data

    def prepare_input(self, example, mode="train"):
        text_a = example.texts[0]
        text_b = example.texts[1]

        inputs = self.tokenizer(text_a,
                                add_special_tokens=True,
                                max_length=self.data_train_args.max_length,
                                padding="max_length",
                                truncation=True,
                                return_offsets_mapping=False,
                                return_tensors="pt")
        inputs['labels'] = label

        return inputs

if __name__ == '__main__':
    TrainSimCSE(config_path="BasicTask/SentenceEmbedding/SimCSE_ST/config.yaml").run()
