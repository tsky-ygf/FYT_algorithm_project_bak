# -*- coding: utf-8 -*-
# @Time    : 2022/09/08 15:24
# @Author  : Czq
# @File    : utils.py
# @Software: PyCharm
import json
import os
import random
import pandas as pd
import numpy as np
from pprint import pprint
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')
# tokenizer = BertTokenizer.from_pretrained('model/language_model/bert-base-chinese')


def read_config(config_path):
    config_data = pd.read_csv(config_path, encoding='utf-8', na_values=' ', keep_default_na=False)
    config_list = []
    label2alias = dict()
    _alias2label = dict()

    for line in config_data.values:
        config_list.append(line[0])
        alis = line[1].split('|')
        if alis:
            for ali in alis:
                if ali:
                    _alias2label[ali] = line[0]
        _alias2label[line[0]] = line[0]
    config_list = list(filter(None, config_list))
    return config_list, _alias2label


# 生成所有的通用label， 包含别称
def read_config_to_label(args):
    config_path = 'data/data_src/config.csv'
    # 读取config，将别称也读为schema
    config_list, _alias2label = read_config(config_path)

    config_list.remove('争议解决')
    config_list.remove('通知与送达')
    config_list.remove('甲方解除合同')
    config_list.remove('乙方解除合同')
    config_list.remove('未尽事宜')
    # config_list.remove('附件')
    config_list.remove('金额')

    # config_list = ['争议解决','合同生效','未尽事宜','通知与送达','鉴于条款','附件']
    # config_list.insert(0, 'O') # ============ for softmax
    return config_list, _alias2label
    # return ['争议解决','合同生效','未尽事宜','通知与送达','鉴于条款','附件'], _alias2label

#
def read_config_to_label_aux():
    config_path = 'data/data_src/config_aux.csv'
    config_data = pd.read_csv(config_path, encoding='utf-8', na_values=' ', keep_default_na=False)
    schemas = config_data['schema'].values.tolist()
    schema2new_schema = dict()
    new_schemas = config_data['new_schema'].values.tolist()
    for sche, ns in zip(schemas, new_schemas):
        schema2new_schema[sche] = ns
    return list(set(new_schemas)), schema2new_schema


# 加载train和dev数据
def load_data(path):
    out_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line.strip())
            # out_data.append([json_line['content'],json_line['result_list'],json_line['prompt']])
            out_data.append(json_line)
    return out_data


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


class ReaderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def batchify_cluener(batch):
    sentences = []
    labels = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    start_seqs = []
    end_seqs = []
    for b in batch:
        content = b['text']
        res_dict = b['label']
        content_l = list(content)
        sentences.append(content)
        input_i = [101] + tokenizer.convert_tokens_to_ids(list(content_l)) + [102]
        input_id = input_i.copy() + [0] * (52 - len(input_i))
        atten_mask = [1] * len(input_id) + [0] * (52 - len(input_id))
        token_type_id = [0] * 52

        assert len(input_id) == 52, len(input_id)
        input_ids.append(input_id)
        attention_mask.append(atten_mask)
        token_type_ids.append(token_type_id)

        # start_seq = [[0] * 50 for _ in range(10)]
        # end_seq = [[0] * 50 for _ in range(10)]
        start_seq = [0] * 50
        end_seq = [0] * 50

        for label, res in res_dict.items():
            for entity, position in res.items():
                # labels.append([label, entity, position[0]])
                labels.append([label, entity])
                # labels.append([label, position[0][0], position[0][1]])
                label_index = labels2id.index(label)
                start, end = position[0][0], position[0][1]
                start_seq[start] = label_index
                end_seq[end] = label_index
        start_seqs.append(start_seq)
        end_seqs.append(end_seq)

    encoded_dict = {
        'input_ids': torch.LongTensor(input_ids).to('cuda'),
        'attention_mask': torch.LongTensor(attention_mask).to('cuda'),
        'token_type_ids': torch.LongTensor(token_type_ids).to('cuda')
    }
    start_seqs = torch.LongTensor(start_seqs).to('cuda')
    end_seqs = torch.LongTensor(end_seqs).to('cuda')
    assert len(input_ids) == len(start_seqs), [len(input_ids), len(start_seqs)]
    return encoded_dict, start_seqs, end_seqs, labels, sentences


def batchify(batch):
    # 在doccano_data_preprocess中，已经做过最大长度截断了。
    sentences = []
    labels = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    window_length = 510  # add 101 102 to 512
    start_seqs = []
    end_seqs = []
    start_seqs_aux = []
    end_seqs_aux = []
    # labels_aux = []

    for b in batch:
        text = b['text']
        res_list = b['entities']
        start_seq = [[0] * window_length for _ in range(len(labels2id))]
        end_seq = [[0] * window_length for _ in range(len(labels2id))]
        start_seq_a = [[0] * window_length for _ in range(len(labels2id_aux))]
        end_seq_a = [[0] * window_length for _ in range(len(labels2id_aux))]
        sentences.append(text)
        if not res_list:
            start_seqs.append(start_seq)
            end_seqs.append(end_seq)
        else:
            for res in res_list:
                label = res['label']
                if label not in labels2id:
                    continue
                start = res['start_offset']
                end = res['end_offset']
                if start is not None and end is not None:
                    entity_text = text[start:end]
                    labels.append([label, entity_text])
                label_id = labels2id.index(label)
                new_label_a = label2new_label[label]
                label_id_a = labels2id_aux.index(new_label_a)
                if start is not None:
                    start_seq[label_id][start] = 1
                    start_seq_a[label_id_a][start] = 1
                if end is not None:
                    end_seq[label_id][end] = 1
                    end_seq_a[label_id_a][end] = 1

            start_seqs.append(start_seq)
            end_seqs.append(end_seq)
            start_seqs_aux.append(start_seq_a)
            end_seqs_aux.append(end_seq_a)

        input_i = [101] + tokenizer.convert_tokens_to_ids(list(text)) + [102]
        input_id = input_i.copy() + [0] * (512 - len(input_i))
        atten_mask = [1] * len(input_i) + [0] * (512 - len(input_i))
        token_type_id = [0] * 512
        assert len(input_id) == 512, len(input_id)

        input_ids.append(input_id)
        attention_mask.append(atten_mask)
        token_type_ids.append(token_type_id)

    encoded_dict = {
        'input_ids': torch.LongTensor(input_ids).to('cuda'),
        'attention_mask': torch.LongTensor(attention_mask).to('cuda'),
        'token_type_ids': torch.LongTensor(token_type_ids).to('cuda')
    }

    # no sense for softmax method
    start_seqs = torch.FloatTensor(start_seqs).transpose(1, 2).to('cuda')
    end_seqs = torch.FloatTensor(end_seqs).transpose(1, 2).to('cuda')
    start_seqs_aux = torch.FloatTensor(start_seqs_aux).transpose(1,2).cuda()
    end_seqs_aux = torch.FloatTensor(end_seqs_aux).transpose(1,2).cuda()

    # same batch
    assert len(input_ids) == len(start_seqs), [len(input_ids), len(start_seqs)]
    return encoded_dict, start_seqs, end_seqs, labels, sentences, [start_seqs_aux, end_seqs_aux]


def evaluate_entity_wo_category(true_entities, pred_entities):
    # 不论实体类别的评估
    rights = [entity for entity in pred_entities if entity in true_entities]
    origin = len(true_entities)
    found = len(pred_entities)
    right = len(rights)
    recall_ent = 0 if origin == 0 else (right / origin)
    precision_ent = 0 if found == 0 else (right / found)
    f1_ent = 0. if recall_ent + precision_ent == 0 else (2 * precision_ent * recall_ent) / (precision_ent + recall_ent)
    print("numbers of true positive", origin)  # 418
    print("numbers of predicted positive", right)
    # print("epoch:", e, "  p: {0}, r: {1}, f1: {2}".format(precision_ent, recall_ent, f1_ent))
    return precision_ent, recall_ent, f1_ent


def evaluate_index(y_pred, y_true):
    # calculate p r f1
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    print("numbers of true positive", np.sum(y_true))  # 418
    print("numbers of predicted positive", np.sum(y_pred))
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S


if __name__ == "__main__":
    # file = 'data/cluener/dev.json'
    # maxlen = 0 # 52
    # new maxlength of entity  263
    pass

else:
    labels2id, alias2label = read_config_to_label(None)
    labels2id_aux, label2new_label = read_config_to_label_aux()