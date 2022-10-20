#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/20 17:36
# @Author  : Adolf
# @Site    : 
# @File    : train.py
# @Software: PyCharm
from BasicTask.Classification.Bert.cls_train import TrainClassification
from Tools.data_pipeline import InputExample

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class QueryClassificationTrain(TrainClassification):

    def create_examples(self, data_path, mode):
        self.logger.info("Creating {} examples".format(mode))
        self.logger.info("Creating examples from {} ".format(data_path))
        examples = []

        with open(data_path, "r") as f:
            for index, line in enumerate(f.readlines()):
                text = line.split("\t")[0]
                label = line.split("\t")[1].strip()
                try:
                    examples.append(InputExample(guid=str(index), texts=[text], label=int(label)))
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.warning(f"index:{index}, text:{text}, label:{label}")
        return examples


if __name__ == '__main__':
    QueryClassificationTrain(config_path="IntelligentConsultation/src/query_classification/train_config.yaml").run()
