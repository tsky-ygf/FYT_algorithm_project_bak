#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 15:27
# @Author  : Adolf
# @Site    :
# @File    : anyou_dataset.py
# @Software: PyCharm
import torch
import json
import torch.utils.data as data


def prepare_input(text, tokenizer, max_len=512):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=False,
        return_tensors="pt",
    )

    return inputs


class LawsAnyouClsDataset(data.Dataset):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        with open(data_path, "rb") as f:
            self.inputs = json.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        pass


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer_path = "model/language_model/chinese-roberta-wwm-ext"
    tokenizer_ = AutoTokenizer.from_pretrained(tokenizer_path)
    data_path_ = "data/fyt_train_use_data/CAIL-Long/civil/dev.json"
    tmp_dataset = LawsAnyouClsDataset(tokenizer_, data_path_)
    item_ = tmp_dataset[10]
    print(item_)
