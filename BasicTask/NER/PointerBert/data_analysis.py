#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/29 15:18
# @Author  : Czq
# @File    : data_analysis.py
# @Software: PyCharm
import json
import os
from collections import defaultdict
from pprint import pprint

import numpy

from DocumentReview.PointerBert.utils import read_config_to_label


def fun1():
    file = 'data/data_src/common_long/dev.json'
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
                    # if entity['start_offset'] is None:
                    #     print(entity)
                    print(entity['label'], ':::', text[entity['start_offset']:entity['end_offset']])
                    print(entity)
                    print('*'*50)


def fun2():
    labels2id, alias2label = read_config_to_label(None)

    print("label number", len(labels2id))
    file_path = 'data/data_src/common_0926'
    data_all = []
    to_file = 'data/data_src/common_all/common_all.json'
    w = open(to_file, 'w', encoding='utf-8')
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join('data/data_src/common_0926', file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                if 'caigou' in file or 'common.jsonl' in file or 'jietiao' in file:
                    labels = line['entities']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab['label'] not in alias2label:
                            continue
                        label = alias2label[lab['label']]
                        if label not in labels2id:
                            continue
                        new_labels[lab['label']].append([lab['start_offset'], lab['end_offset']])
                    for lab, indexs in new_labels.items():
                        if len(indexs) == 1:
                            continue
                        indexs.sort()
                        for ii in range(1, len(indexs)):
                            if indexs[ii][0] == indexs[ii][1]:
                                print("there are", new_labels)
                                print("id:", idd)
                                print("text ", text)

                else:
                    labels = line['label']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab[2] not in alias2label:
                            continue
                        label = alias2label[lab[2]]
                        if label not in labels2id:
                            continue
                        new_labels[label].append([lab[0],lab[1]])

                    for lab, indexs in new_labels.items():
                        if len(indexs) == 1:
                            continue
                        indexs.sort()
                        for ii in range(1, len(indexs)):
                            if indexs[ii-1][1] == indexs[ii][0]:
                                print("there are", new_labels)
                                print("id:", idd)
                                print("text ", text)


def fun3():
    labels2id, alias2label = read_config_to_label(None)
    labels2id.append('争议解决')
    labels2id.append('通知与送达')
    labels2id.append('未尽事宜')
    labels2id.append('附件')
    print("label number", len(labels2id))
    file_path = 'data/data_src/common_0926'
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join('data/data_src/common_0926', file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                dp = numpy.zeros(len(text))
                if 'caigou' in file or 'common.jsonl' in file or 'jietiao' in file:
                    labels = line['entities']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab['label'] not in alias2label:
                            continue
                        label = alias2label[lab['label']]
                        if label not in labels2id:
                            continue
                        new_labels[lab['label']].append([lab['start_offset'], lab['end_offset']])
                        dp[lab['start_offset']:lab['end_offset']] +=1
                else:
                    labels = line['label']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab[2] not in alias2label:
                            continue
                        label = alias2label[lab[2]]
                        if label not in labels2id:
                            continue
                        new_labels[lab[2]].append([lab[0],lab[1]])
                        dp[lab[0]:lab[1]] += 1
                if 2 in dp:
                    print(idd, text, new_labels)
                    for l, v in new_labels.items():
                        for _ in v:
                            print(l, text[_[0]:_[1]])





if __name__ == "__main__":
    # print("wefwefwfe"[1:4])
    # print("wefwefwfe"[1:None]) # efwefwfe
    fun1()  # 查看标注
    # fun2()  # 验证连续的标签
    # fun3() # 检查重叠
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