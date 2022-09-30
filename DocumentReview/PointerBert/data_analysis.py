#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/29 15:18
# @Author  : Czq
# @File    : data_analysis.py
# @Software: PyCharm
import json
from pprint import pprint


def fun1():
    file = 'data/data_src/common_long/train.json'
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            # if i == 3:
                j = json.loads(line)
                text = j['text']
                entities = j['entities']
                print(text)
                print(j['id'])
                print('-'*100)

                for entity in entities:
                    if entity['start_offset'] is None:
                        print(entity)
                    # print(entity['label'], ':::', text[entity['start_offset']:entity['end_offset']])
                    # print(entity)
                        print('*'*50)

if __name__ == "__main__":
    # print("wefwefwfe"[1:4])
    # print("wefwefwfe"[1:None]) # efwefwfe
    fun1()
    # from transformers import AutoTokenizer
    # t = AutoTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')
    # print(len("今天天气是阴天，明天是多云，后天是晴天"))
    #     # res = t("今天天气是阴天，明天是多云，后天是晴天",
    #     #         truncation=True,
    #     #         max_length=12,
    #     #         stride=5,
    #     #         return_overflowing_tokens=True,
    #     #         return_offsets_mapping=True,
    #     #         padding="max_length")
    # print(len(res['input_ids'][0]), len(res['input_ids'][1]),len(res['input_ids'][2]))
    # print(res)

    pass