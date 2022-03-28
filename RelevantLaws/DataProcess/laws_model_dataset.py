#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 17:36
# @Author  : Adolf
# @Site    : 
# @File    : laws_model_dataset.py
# @Software: PyCharm
import torch
import json
import pandas as pd
import torch.utils.data as data


def prepare_input(tokenizer, text):
    inputs = tokenizer(text,
                       add_special_tokens=True,
                       max_length=512,
                       padding="max_length",
                       truncation=True,
                       return_offsets_mapping=False,
                       return_tensors="pt")

    return inputs


class LawsThuNLPDataset(data.Dataset):
    def __init__(self, tokenizer, data_path, mapping_path, logger):
        self.logger = logger
        self.tokenizer = tokenizer
        with open(data_path, 'rb') as f:
            self.inputs = json.load(f)

        map_df = pd.read_csv(mapping_path)
        self.label_map = dict(zip(map_df['laws'], map_df['index']))
        self.num_labels = len(self.label_map)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        # print(len(self.inputs[item]["fact"]))
        # print(self.inputs[item]['laws'])
        # print(self.label_map)
        label = []

        for one_law in self.inputs[item]['laws']:
            if "诉讼" not in one_law[0]['title']:
                label.append(one_law[0]['title'] + '###' + one_law[1])

        inputs = prepare_input(self.tokenizer,
                               self.inputs[item]["fact"])

        label = [self.label_map[one] for one in label]
        label = torch.tensor(label)

        try:
            y_onehot = torch.nn.functional.one_hot(label, num_classes=self.num_labels)
            y_onehot = y_onehot.sum(dim=0).float()
        except Exception as e:
            # print(e)
            self.logger.debug(e)
            y_onehot = torch.tensor([0.0] * self.num_labels)

        inputs["labels"] = y_onehot

        return inputs


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer_path = "model/language_model/chinese-roberta-wwm-ext"
    tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)
    data_path_ = "data/fyt_train_use_data/CAIL-Long/civil/dev.json"
    mapping_path_ = "data/fyt_train_use_data/CAIL-Long/civil/label_mapping.csv"
    tmp_dataset = LawsThuNLPDataset(tokenizer_, data_path_, mapping_path_)
    item_ = tmp_dataset[10]
    print(item_)
