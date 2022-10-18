#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/08 16:14
# @Author  : Czq
# @File    : doccano_data_preprocess.py
# @Software: PyCharm
import json
import os

import numpy

from BasicTask.NER.PointerBertMRC.main import read_config_to_label

def merge_all_data4common():
    # list      dict
    labels2id, alias2label = read_config_to_label(None)
    print("label number", len(labels2id))
    file_path = 'data/data_src/common_1008'
    data_all = []
    to_file = 'data/data_src/common_mrc/common_mrc.json'
    w = open(to_file, 'w', encoding='utf-8')
    window = 510-15
    # unused = []
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join(file_path,file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                if 'caigou' in file or 'common.jsonl' in file or 'jietiao' in file:
                    labels = line['entities']
                    flag_d = True
                else:
                    labels = line['label']
                    flag_d = False
                # 截断， 滑动窗口window， step：400
                for i in range(0, len(text), 400):
                    entities_new = []
                    bias = i
                    text_split = text[i:i + window]
                    for entity in labels:
                        if flag_d:
                            if entity['label'] not in alias2label:
                                # unused.append(entity['label'])
                                continue
                            label = alias2label[entity['label']]
                            if label not in labels2id:
                                continue
                            # 若截断，也保留单个start或end
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
                                # unused.append(entity[2])
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
    # unused = set(unused)
    # print(unused)
    pass


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
    data = numpy.array(data)
    train_data = data[train_index]
    dev_data = data[dev_index]

    with open(os.path.join(to_path,'train.json'), 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line)
    with open(os.path.join(to_path, 'dev.json'),'w', encoding='utf-8') as f:
        for line in dev_data:
            f.write(line)

if __name__ == "__main__":
    merge_all_data4common()
    divided_train_dev('data/data_src/common_mrc/common_mrc.json', 'data/data_src/common_mrc/')