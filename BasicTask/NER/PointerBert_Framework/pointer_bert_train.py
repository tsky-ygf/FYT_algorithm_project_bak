#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 15:15
# @Author  : Czq
# @File    : pointer_bert_train.py
# @Software: PyCharm
import json
import os

from transformers import BertTokenizer

from BasicTask.NER.PointerBert_Framework.model_NER import PointerNERBERTInFramework
from Tools.data_pipeline import InputExample
from Tools.train_tool import BaseTrainTool
from BasicTask.NER.PointerBert_Framework.utils import read_config_to_label

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class TrainPointerBert(BaseTrainTool):
    def __init__(self, config_schema, is_long, *args, **kwargs):
        self.schema2id, self.num_labels = read_config_to_label(config_schema, is_long=is_long)
        super().__init__(*args, **kwargs)

    def init_model(self, *args, **kwargs):
        tokenizer = BertTokenizer.from_pretrained(self.model_args.tokenizer_name)
        self.model_args.mode = self.model_args.model_name_or_path
        self.model_args.num_labels = self.num_labels
        model = PointerNERBERTInFramework(self.model_args)
        return tokenizer, model

    def create_examples(self, data_path, mode="train"):
        self.logger.info("Creating {} examples".format(mode))
        self.logger.info("Creating examples from {} ".format(data_path))

        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                for entity in line['entities']:
                    entity['label'] = self.schema2id[entity['label']]
                examples.append(InputExample(guid=line["id"], texts=[line["text"]], label=line['entities']))

        return examples

    # def prepare_input(self, example, mode="train"):
    #     pass




if __name__ == "__main__":
    t = TrainPointerBert(config_path="BasicTask/NER/PointerBert_Framework/base_p.yaml",
                         config_schema='data/data_src/config.csv', is_long=False)
    # t.run()
    pass
