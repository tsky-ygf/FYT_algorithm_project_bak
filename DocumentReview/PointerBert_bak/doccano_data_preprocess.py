#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/22 13:29
# @Author  : Czq
# @File    : doccano_data_preprocess.py
# @Software: PyCharm
import json
import os
from collections import defaultdict
from pprint import pprint

import numpy
import numpy as np
import torch
from DocumentReview.PointerBert.utils import read_config_to_label


def split_text(file, to_file):
    w = open(to_file, 'w', encoding='utf-8')

    window = 510
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            text = line['text'].replace('\xa0',' ')
            used = []
            entities = line['entities']
            for i in range(0, len(text), 400):
                entities_new = []
                bias = i
                text_split = text[i:i+window]
                for entity in entities:
                    if i<= entity['start_offset']<entity['end_offset']<i+window:
                        entities_new.append({'label':entity['label'],
                                             'start_offset':entity['start_offset']-bias, 'end_offset':entity['end_offset']-bias})
                        used.append(entity)
                w.write(json.dumps({'id':line['id'],
                                    'text': text_split,
                                    'entities': entities_new
                                    },ensure_ascii=False)+'\n')

            if len(entities) > len(used):
                print('true', len(entities))
                print('used', len(used))
                print(line)
    w.close()


def divided_train_dev(file, to_path):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    l = len(data)
    print("samples number", l)
    arr_index = list(range(l))
    numpy.random.shuffle(arr_index)

    p = int(l*0.81)
    train_index = arr_index[:p]
    dev_index = arr_index[p:]
    print("train dataset number", len(train_index))
    print("dev dataset number", len(dev_index))
    data = np.array(data)
    train_data = data[train_index]
    dev_data = data[dev_index]

    with open(os.path.join(to_path,'train.json'), 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line)
    with open(os.path.join(to_path, 'dev.json'),'w', encoding='utf-8') as f:
        for line in dev_data:
            f.write(line)

# 转换为cluener的数据格式
def convert_format(in_file, out_file):

    w = open(out_file, 'w', encoding='utf-8')

    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            text = line['text']
            entities = line['entities']
            labels = dict()
            for entity in entities:
                entity_text = text[entity['start_offset']:entity['end_offset']]
                labels[entity['label']] = {entity_text: [[entity['start_offset'], entity['end_offset']-1]]}

            res = {'text':text, 'label':labels}
            w.write(json.dumps(res, ensure_ascii=False)+"\n")

    w.close()

# 通用条款， 融合所有数据
def merge_all_data4common():
    # list      dict
    labels2id, alias2label = read_config_to_label(None)

    print("label number", len(labels2id))
    file_path = 'data/data_src/common_0926'
    data_all = []
    to_file = 'data/data_src/common_all/common_all.json'
    w = open(to_file, 'w', encoding='utf-8')
    window = 510
    unused = []
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join('data/data_src/common_0926',file)
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
                        new_indexs = []
                        for ii in range(1, len(indexs)):
                            if indexs[ii][0] == indexs[ii-1][1]:
                                new_indexs.append([indexs[ii-1][0], indexs[ii][1]])
                            else:
                                new_indexs.append(indexs[ii-1].copy())
                        new_indexs.append(indexs[-1].copy())
                        new_labels[lab] = new_indexs



                    flag_d = True
                else:
                    labels = line['label']
                    flag_d = False
                for i in range(0, len(text), 400):
                    entities_new = []
                    bias = i
                    text_split = text[i:i + window]
                    # 增加一步， 有相同label的实体， 若连续的， 则合并。
                    for entity in labels:
                        if flag_d:
                            if entity['label'] not in alias2label:
                                unused.append(entity['label'])
                                continue
                            label = alias2label[entity['label']]
                            if label not in labels2id:
                                continue
                            entity_new = {'label': None, 'start_offset': None,
                                                     'end_offset': None}
                            if i<= entity['start_offset']<i+window:
                                entity_new['label'] = label
                                entity_new['start_offset'] = entity['start_offset'] - bias
                            if i<=entity['end_offset']<i+window:
                                entity_new['label'] = label
                                entity_new['end_offset'] = entity['end_offset'] - bias
                            if entity_new['label'] is not None:
                                entities_new.append(entity_new)
                            # if i <= entity['start_offset'] < entity['end_offset'] < i + window:
                            #     entities_new.append({'label': label,
                            #                          'start_offset': entity['start_offset'] - bias,
                            #                          'end_offset': entity['end_offset'] - bias})
                        else:
                            if entity[2] not in alias2label:
                                unused.append(entity[2])
                                continue
                            label = alias2label[entity[2]]
                            if label not in labels2id:
                                continue
                            entity_new = {'label': None, 'start_offset': None,
                                          'end_offset': None}
                            if i <= entity[0] < i + window:
                                entity_new['label'] = label
                                entity_new['start_offset'] = entity[0] - bias
                            if i <= entity[1] < i + window:
                                entity_new['label'] = label
                                entity_new['end_offset'] = entity[1] - bias
                            if entity_new['label'] is not None:
                                entities_new.append(entity_new)
                            # if i <= entity[0] < entity[1] < i + window:
                            #     entities_new.append({'label': label,
                            #                          'start_offset': entity[0] - bias,
                            #                          'end_offset': entity[1] - bias})
                    # 没有负例
                    if entities_new:
                        w.write(json.dumps({'id': idd,
                                            'text': text_split,
                                            'entities': entities_new
                                            }, ensure_ascii=False) + '\n')
    w.close()
    unused = set(unused)
    print(unused)
    pass


if __name__ == "__main__":
    # split_text('data/data_src/old/origin_oldall.json', 'data/data_src/old/origin_oldall_split.json')
    # t()
    # convert_format('data/data_src/new/dev_300.json', 'data/data_src/cluener_format/dev_300.json')
    # convert_format('data/data_src/new/train_300.json', 'data/data_src/cluener_format/train_300.json')
    # convert_format_bmes()

    # 先这两条， 再改log、model名称， 再运行
    merge_all_data4common()
    divided_train_dev('data/data_src/common_all/common_all.json', 'data/data_src/common_all/')
    pass