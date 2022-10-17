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
    # print("wefwefwfe"[1:4])
    # print("wefwefwfe"[1:None]) # efwefwfe
    # fun1()  # 查看标注
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

    # uie result
    # t = [0.56731,0.58847,0.68158,0.59454,0.76901,0.72313,0.74951,0.65434,0.86831,0.78641]
    # print(sum(t))

#     text = """第二十一条乙方不办理或者拖延办理工作交接和离职手续，或者因乙方责任造成甲方的法律（诉讼）纠纷损害甲方利益的，应当承担由此而产生的一切法律责任。
# 第二十二条乙方在本合同首页填写的通知送达地发生变化，应在一周内以书面形式通知甲方。如乙方未按规定及时通知甲方，甲方寄送的有关文书到达上述通知送达地，即视为送达，由乙方承担后果和相应的法律责任。
# 第二十三条双方因履行本合同发生争议，当事人可以向甲方劳动争议调解委员会申请调解；调解不成的，可以向劳动争议仲裁委员会申请仲裁。当事人一方也可以直接向劳动争议仲裁委员会申请仲裁。
# 第二十四条本合同一式两份，甲乙双方各执一份。
# 甲方：（盖章）乙方：（签名）
# 法定代表人或（委托代理人）：
# 2022年1月1日2022年1月1日"""
#     print(len(text))
    tokenizer = BertTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')
    # inputs = tokenizer(text,
    #                         add_special_tokens=True,
    #                         max_length=512,
    #                         padding="max_length",
    #                         truncation=True,
    #                         return_offsets_mapping=False,
    #                         return_tensors="pt")
    # print(sum(inputs['attention_mask'][0]))
    #
    # print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    #
    # token_ids = tokenizer.convert_tokens_to_ids(list(text))
    # print(len(token_ids))
    # print(list(text))
    # print('----')
    # print(token_ids)

    tokenizer.add_special_tokens({'additional_special_tokens':["\n", " "]})
    text = "2022年我赚了23423425325"
    inputs = tokenizer(text,
                       add_special_tokens=True,
                       max_length=512,
                       padding="max_length",
                       truncation=True,
                       return_offsets_mapping=False,
                       return_tensors="pt")
    tokenizer.convert_tokens_to_ids()
    print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
    print(len(text), sum(inputs['attention_mask'][0]))
    print(list(text))
    pass