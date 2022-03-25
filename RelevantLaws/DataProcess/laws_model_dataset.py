#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 17:36
# @Author  : Adolf
# @Site    : 
# @File    : laws_model_dataset.py
# @Software: PyCharm
import torch
import json
import torch.utils.data as data


def prepare_input(tokenizer, text):
    inputs = tokenizer(text,
                       add_special_tokens=True,
                       max_length=512,
                       padding="max_length",
                       return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class LawsThuNLPDataset(data.Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        with open(data_path, 'rb') as f:
            self.inputs = json.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        # print(len(self.inputs[item]["fact"]))
        print(self.inputs[item]['laws'])
        label = []

        for one_law in self.inputs[item]['laws']:
            label.append(one_law[0]['title'] + '###' + one_law[1])
        print(label)

        inputs = prepare_input(self.tokenizer,
                               self.inputs[item]["fact"])

        return inputs, label


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer_path = "model/language_model/chinese-roberta-wwm-ext"
    tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)
    data_path_ = "data/fyt_train_use_data/CAIL-Long/civil/dev.json"
    tmp_dataset = LawsThuNLPDataset(tokenizer_, data_path_)
    tmp_dataset[10]
