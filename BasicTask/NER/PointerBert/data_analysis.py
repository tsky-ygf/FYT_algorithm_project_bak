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
from transformers import BertTokenizer

from BasicTask.NER.PointerBert.utils import read_config_to_label
from DocumentReview.src.ParseFile import read_docx_file


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
    file_path = 'data/data_src/common_1013'
    data_all = []
    to_file = 'data/data_src/common_all/common_all.json'
    w = open(to_file, 'w', encoding='utf-8')
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join('data/data_src/common_1013', file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                if 'entities' in line:
                    labels = line['entities']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab['label'] not in alias2label:
                            continue
                        label = alias2label[lab['label']]
                        if label not in labels2id:
                            continue
                        if lab['end_offset']-lab['start_offset']>500:
                            print('-'*50)
                            print('id',idd)
                            print(text)
                            print(lab)

                        # new_labels[lab['label']].append([lab['start_offset'], lab['end_offset']])
                    # for lab, indexs in new_labels.items():
                    #     if len(indexs) == 1:
                    #         continue
                    #     indexs.sort()
                    #     for ii in range(1, len(indexs)):
                    #         if indexs[ii][0] == indexs[ii][1]:
                    #             print("there are", new_labels)
                    #             print("id:", idd)
                    #             print("text ", text)

                else:
                    labels = line['label']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab[2] not in alias2label:
                            continue
                        label = alias2label[lab[2]]
                        if label not in labels2id:
                            continue
                        if lab[1]-lab[0]>500:
                            print('-'*50)
                            print(line['id'])
                            print(line['text'])
                            print(lab)

                    #     new_labels[label].append([lab[0],lab[1]])
                    # for lab, indexs in new_labels.items():
                    #     if len(indexs) == 1:
                    #         continue
                    #     indexs.sort()
                    #     for ii in range(1, len(indexs)):
                    #         if indexs[ii-1][1] == indexs[ii][0]:
                    #             print("there are", new_labels)
                    #             print("id:", idd)
                    #             print("text ", text)


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
    # fun1()  # 查看标注
    # fun2()  # 验证连续的标签
    # fun3() # 检查重叠
    def a_new_decorator(a_func):

        def wrapTheFunction():
            print("I am doing some boring work before executing a_func()")

            a_func()

            print("I am doing some boring work after executing a_func()")

        return wrapTheFunction


    @a_new_decorator
    def a_function_requiring_decoration():
        """Hey you! Decorate me!"""
        print("I am the function which needs some decoration to "
              "remove my foul smell")


    a_function_requiring_decoration()
    print(a_function_requiring_decoration.__name__)

    pass