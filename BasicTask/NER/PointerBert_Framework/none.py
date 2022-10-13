#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 16:03
# @Author  : Czq
# @File    : none.py
# @Software: PyCharm
import os

from transformers import BertTokenizer

if __name__ == "__main__":
    print(os.path.exists('model/language_model/chinese-roberta-wwm-ext'))
    text = '今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球。今天是晴天，足球'
    print(len(text))
    tokenizer = BertTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')
    inputs = tokenizer(text,
                                add_special_tokens=True,
                                max_length=512,
                                padding="max_length",
                                truncation=True,
                                return_offsets_mapping=False,
                                return_tensors="pt")
    print(len(inputs['input_ids'][0]))
    print(inputs['input_ids'][0])

    pass
