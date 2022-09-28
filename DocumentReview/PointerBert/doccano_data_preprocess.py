#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/22 13:29
# @Author  : Czq
# @File    : doccano_data_preprocess.py
# @Software: PyCharm
import json
import os
from pprint import pprint


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
    p = int(l*0.81)
    train_data = data[:p]
    dev_data = data[p:]

    with open(os.path.join(to_path,'train_split.json'), 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line)
    with open(os.path.join(to_path, 'dev_split.json'),'w', encoding='utf-8') as f:
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


def convert_format_bmes():
    labels = ['Title', 'JIA', '']
    pass

if __name__ == "__main__":
    # split_text('data/data_src/old/origin_oldall.json', 'data/data_src/old/origin_oldall_split.json')
    # divided_train_dev('data/data_src/old/origin_oldall_split.json', 'data/data_src/old/')
    # t()
    # convert_format('data/data_src/new/dev_300.json', 'data/data_src/cluener_format/dev_300.json')
    # convert_format('data/data_src/new/train_300.json', 'data/data_src/cluener_format/train_300.json')
    # convert_format_bmes()
    pass
