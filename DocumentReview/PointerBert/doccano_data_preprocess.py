#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/22 13:29
# @Author  : Czq
# @File    : doccano_data_preprocess.py
# @Software: PyCharm
import json
import os


def split_text(file, to_file):
    w = open(to_file, 'w', encoding='utf-8')

    window = 510
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            text = line['text']
            used = []
            entities = line['entities']
            for i in range(0, len(text), 400):
                entities_new = []
                bias = i
                text_split = text[i:i+window]
                for entity in entities:
                    if i<= entity['start_offset']<entity['end_offset']<i+window:
                        entities_new.append({'id': entity['id'], 'label':entity['label'],
                                             'start_offset':entity['start_offset']-bias, 'end_offset':entity['end_offset']-bias})
                        used.append(entity)
                w.write(json.dumps({'id':line['id'],
                                    'text': text_split,
                                    'entities': entities_new
                                    },ensure_ascii=False)+'\n')
            print('true', len(entities))
            print('used', len(used))
            if len(entities) > len(used):
                print(line)
    w.close()


def divided_train_dev(file, to_path):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    l = len(data)
    p = int(l*0.81)
    train_data = data[:p]
    dev_data = data[p:]

    with open(os.path.join(to_path,'train_100.json'), 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line)
    with open(os.path.join(to_path, 'dev_100.json'),'w', encoding='utf-8') as f:
        for line in dev_data:
            f.write(line)

# 转换为cluener的数据格式
def convert_format():
    in_file= 'data/data_src/new/dev_100.json'

    out_file = 'data/data_src/cluener_format/dev_100.json'

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


if __name__ == "__main__":
    # split_text('data/data_src/new/all_100.jsonl', 'data/data_src/new/origin_splitted.json')
    # divided_train_dev('data/data_src/new/origin_splitted.json', 'data/data_src/new/')
    convert_format()