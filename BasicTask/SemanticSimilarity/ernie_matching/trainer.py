#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 15:08
# @Author  : Adolf
# @Site    : 
# @File    : relevant_law.py
# @Software: PyCharm
import pandas as pd
import torch

from Tools.train_tool import BaseTrainTool

from transformers import AutoTokenizer, AutoModel
from BasicTask.SemanticSimilarity.ernie_matching.model import PairwiseMatching

from Tools.data_pipeline import InputExample, DataProcessor, BaseDataset


class ErnieMatchingInputExample(InputExample):
    def __init__(self, guid, text, text_q, subject):
        super().__init__(guid, text, subject)
        self.text_q = text_q


class ErnieMatchingProcessor(DataProcessor):
    @staticmethod
    def read_data(data_dir):
        data = pd.read_csv(data_dir, sep='\t')
        return data

    @staticmethod
    def create_examples(df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for index, row in df.iterrows():
            guid = "%s-%s" % (set_type, index)
            text = row['query']
            text_q = row['title']
            subject = row['neg_title']
            examples.append(ErnieMatchingInputExample(guid=guid, text=text, text_q=text_q, subject=subject))
        return examples


# ErnieMatchingProcessor().get_test_examples(data_dir="data/sort/dev_pairwise.csv")


class ErnieMatchingDataset(BaseDataset):
    @staticmethod
    def prepare_input(example, tokenizer, max_len=512):
        text = example.text
        text_q = example.text_q
        label = example.subject

        pos_inputs = tokenizer(text,
                               add_special_tokens=True,
                               max_length=max_len,
                               padding="max_length",
                               truncation=True,
                               return_offsets_mapping=False,
                               return_tensors="pt")

        neg_inputs = tokenizer(text_q,
                               add_special_tokens=True,
                               max_length=max_len,
                               padding="max_length",
                               truncation=True,
                               return_offsets_mapping=False,
                               return_tensors="pt")

        return {"pos_inputs": pos_inputs, "neg_inputs": neg_inputs, "label": label}


class TrainErnieMatching(BaseTrainTool):
    def __init__(self, config_path):
        super(TrainErnieMatching, self).__init__(config_path=config_path)
        self.logger.info(self.config)

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config['pre_train_tokenizer'])
        pretrained_model = AutoModel.from_pretrained(self.config['pre_train_model'])
        model = PairwiseMatching(pretrained_model)
        return tokenizer, model

    def init_dataset(self):
        data_dir_dict = {'train': self.config['train_data_dir'],
                         'dev': self.config['dev_data_dir']}

        custom_dataset = ErnieMatchingDataset(data_dir_dict=data_dir_dict, tokenizer=self.tokenizer,
                                              mode='train', max_length=128, processor=ErnieMatchingProcessor())
        # dev_dataset = ErnieMatchingDataset(data_dir_dict=data_dir_dict, tokenizer=self.tokenizer,
        #                                    mode='dev', max_length=128, processor=ErnieMatchingProcessor())
        train_size = int(len(custom_dataset) * 0.7)
        test_size = len(custom_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

        return train_dataset, test_dataset

    def cal_loss(self, batch):
        self.logger.debug(batch)
        outputs = self.model(**batch)
        loss = outputs
        return loss


if __name__ == '__main__':
    TrainErnieMatching(config_path="BasicTask/SemanticSimilarity/ernie_matching/config.yaml").train_main()
