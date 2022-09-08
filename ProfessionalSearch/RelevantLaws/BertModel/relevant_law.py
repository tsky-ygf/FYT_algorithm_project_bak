#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/8 16:09
# @Author  : Adolf
# @Site    : 
# @File    : relevant_law.py
# @Software: PyCharm
import json
import pandas as pd
from BasicTask.Classification.Bert.cls_train import TrainClassification
from BasicTask.Classification.Bert.cls_infer import INferClassification

import torch
from Tools.data_pipeline import InputExample


def create_examples(data_path, set_tpye):
    map_df = pd.read_csv("data/fyt_train_use_data/CAIL-Long/civil/label_mapping.csv")
    label_map = dict(zip(map_df['laws'], map_df['index']))
    num_labels = len(label_map)

    examples = []
    with open(data_path, 'rb') as f:
        for (i, example) in enumerate(json.load(f)):
            guid = "%s-%s" % (set_tpye, i)
            text = example['fact']
            label = []

            for one_law in example['laws']:
                if "诉讼" not in one_law[0]['title']:
                    label.append(one_law[0]['title'] + '###' + one_law[1])

            label = [label_map[one] for one in label]
            label = torch.tensor(label)

            try:
                y_onehot = torch.nn.functional.one_hot(label, num_classes=num_labels)
                y_onehot = y_onehot.sum(dim=0).float()
                y_onehot = y_onehot.tolist()
            except Exception as e:
                print(e)
                y_onehot = [0.0] * num_labels

            examples.append(InputExample(guid=guid, text=text, label=[y_onehot]))
            # print(example)
    return examples


class LawsClsTrainer(TrainClassification):
    def __init__(self, config):
        super(LawsClsTrainer, self).__init__(config, create_examples=create_examples)


class LawsClsInfer(INferClassification):
    def __init__(self, config):
        super(LawsClsInfer, self).__init__(config, create_examples=create_examples)


if __name__ == '__main__':
    # LawsClsTrainer(config="RelevantLaws/BertModel/config.yaml").run()
    LawsClsInfer(config="RelevantLaws/BertModel/infer_config.yaml").run()
