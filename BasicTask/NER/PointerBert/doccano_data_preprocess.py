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
from BasicTask.NER.PointerBert.utils import  read_config_to_label


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
    # for long
    # labels2id = ['争议解决','合同生效','未尽事宜','通知与送达','鉴于条款','附件']
    print("label number", len(labels2id))
    file_path = 'data/data_src/common_1013'
    data_all = []
    to_file = 'data/data_src/common_all/common_all.json'
    w = open(to_file, 'w', encoding='utf-8')
    window = 510
    # unused = []
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join(file_path,file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                if 'entities' in line:
                    labels = line['entities']
                    flag_d = True
                else:
                    labels = line['label']
                    flag_d = False
                # 截断， 滑动窗口window， step：400
                for i in range(0, len(text), 400): # len(text)-window, 400)
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


def merge_all_data4common_wt_bias():
    """
    融合所有标注样本 生成用于通用模型训练的数据
    在滑动窗口切分时，保留bias
    """


def merge_all_data4common_wt_cat():
    """
    融合所有标注样本 生成用于通用模型训练的数据
    首尾截取256 拼接
    """
    labels2id, alias2label = read_config_to_label(None)
    print("label number", len(labels2id))
    file_path = 'data/data_src/common_1013'
    to_file = 'data/data_src/common_all_cat/common_all_cat_head_tail.json'
    w = open(to_file, 'w', encoding='utf-8')
    window = 510
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join(file_path,file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                # 当长度小于window
                if len(text)<=window:
                    if 'entities' in line:
                        new_entities = []
                        for entity in line['entities']:
                            if entity['label'] not in alias2label:
                                continue
                            new_label = alias2label[entity['label']]
                            if new_label not in labels2id:
                                continue
                            new_entities.append({'label':new_label,
                                                 'start_offset':entity['start_offset'],
                                                 'end_offset':entity['end_offset']})
                    else:
                        labels = line['label']
                        new_entities = []
                        for label in labels:
                            if label[2] not in alias2label:
                                continue
                            new_label = alias2label[label[2]]
                            if new_label not in labels2id:
                                continue
                            new_entities.append({'label':new_label,
                                                 'start_offset':label[0],
                                                 'end_offset':label[1]})
                    w.write(json.dumps({'id':idd,
                                        'text':text,
                                        'entities':new_entities}, ensure_ascii=False)+"\n")
                # 当长度大于window，先首尾拼接, 再滑动
                else:
                    if 'label' in line:
                        entities = []
                        for label in line['label']:
                            entities.append({'label':label[2],
                                                 'start_offset':label[0],
                                                 'end_offset':label[1]})
                    else:
                        entities = line['entities']

                    new_entities = []
                    text_split = text[:window//2]+"\n"+text[-window//2+1:]
                    assert len(text_split) == window
                    for entity in entities:
                        if entity['label'] not in alias2label:
                            continue
                        new_label = alias2label[entity['label']]
                        if new_label not in labels2id:
                            continue
                        entity_new = {'label': None, 'start_offset': None,
                                      'end_offset': None}
                        if 0 <= entity['start_offset'] < window//2:
                            entity_new['start_offset'] = entity['start_offset']
                            entity_new['label'] = new_label
                        if 0 <= entity['end_offset'] < window//2:
                            entity_new['end_offset'] = entity['end_offset']
                            entity_new['label'] = new_label
                        if entity_new['label']:
                            new_entities.append(entity_new)

                        entity_new = {'label': None, 'start_offset': None,
                                      'end_offset': None}
                        if len(text)-window//2+1 <= entity['start_offset'] < len(text):
                            entity_new['start_offset'] = entity['start_offset']+window-len(text)-1
                            entity_new['label'] = new_label
                        if len(text)-window//2+1 <= entity['start_offset'] < len(text):
                            entity_new['end_offset'] = entity['end_offset']+window-len(text)-1
                            entity_new['label'] = new_label
                        if entity_new['label']:
                            new_entities.append(entity_new)

                    if new_entities:
                        w.write(json.dumps({'id': idd,
                                            'text': text_split,
                                            'entities': new_entities
                                            }, ensure_ascii=False) + '\n')

                    # 去掉首尾， 但是保留一点滑动窗口的重复
                    text = text[:-window//2+1+50]
                    for i in range(window//2-50, len(text)-window//2+1, 400):
                        new_entities = []
                        bias = i
                        text_split = text[i:i + window]
                        for entity in entities:
                            if entity['label'] not in alias2label:
                                continue
                            new_label = alias2label[entity['label']]
                            if new_label not in labels2id:
                                continue
                            entity_new = {'label': None, 'start_offset': None,
                                          'end_offset': None}
                            if i <= entity['start_offset'] < i + window:
                                entity_new['label'] = new_label
                                entity_new['start_offset'] = entity['start_offset'] - bias
                            if i <= entity['end_offset'] < i + window:
                                entity_new['label'] = new_label
                                entity_new['end_offset'] = entity['end_offset'] - bias
                            if entity_new['label'] is not None:
                                new_entities.append(entity_new)

                        if new_entities:
                            w.write(json.dumps({'id': idd,
                                                'text': text_split,
                                                'entities': new_entities
                                                }, ensure_ascii=False) + '\n')
    w.close()
    pass


if __name__ == "__main__":
    # split_text('data/data_src/old/origin_oldall.json', 'data/data_src/old/origin_oldall_split.json')
    # t()
    # convert_format('data/data_src/new/dev_300.json', 'data/data_src/cluener_format/dev_300.json')
    # convert_format('data/data_src/new/train_300.json', 'data/data_src/cluener_format/train_300.json')
    # convert_format_bmes()

    # 先这两条，准备好数据， 再核实schema， 再更改log、model名称， 再运行
    # merge_all_data4common()
    # divided_train_dev('data/data_src/common_all/common_all.json', 'data/data_src/common_all/')

    # merge_all_data4common()
    # divided_train_dev('data/data_src/common_long/common_long.json', 'data/data_src/common_long')

    # merge_all_data4common_wt_bias()

    merge_all_data4common_wt_cat()
    divided_train_dev('data/data_src/common_all_cat/common_all_cat_head_tail.json', 'data/data_src/common_all_cat/')


    pass