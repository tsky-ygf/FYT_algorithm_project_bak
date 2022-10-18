#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/8 16:09
# @Author  :
# @Site    : 
# @File    : relevant_law.py
# @Software: PyCharm
import json

import datasets
import math
import pandas as pd
from tqdm import tqdm

from BasicTask.Classification.Bert.cls_train import TrainClassification
from BasicTask.Classification.Bert.cls_infer import INferClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Tools.data_pipeline import InputExample
from sklearn.metrics import classification_report

from metrics import Metric

def create_examples(data_path, set_tpye):
    map_df = pd.read_csv("data/CAIL/2018/CAIL2018_ALL_DATA/final_all_data/exercise_contest/label_mapping.csv")
    label_map = dict(zip(map_df['label'], map_df['index']))
    num_labels = len(label_map)

    examples = []
    index_i = 0
    with open(data_path, 'rb') as f:
        for item in tqdm(f):
            example = json.loads(item.strip())
            guid = "%s-%s" % (set_tpye, index_i)
            text = example['fact']
            label = example['meta']['accusation']

            label = [label_map[one] for one in label]
            label = torch.tensor(label)

            try:
                y_onehot = torch.nn.functional.one_hot(label, num_classes=num_labels)
                y_onehot = y_onehot.sum(dim=0).float()
                for index_y, item_y in enumerate(y_onehot):
                    if item_y > float(1):
                        y_onehot[index_y] = float(1)
                y_onehot = y_onehot.tolist()
            except Exception as e:
                print(e)
                y_onehot = [0.0] * num_labels

            examples.append(InputExample(guid=guid, text=text, label=[y_onehot]))
            index_i += 1
            # print(example)
    return examples

def label_emb():
    return 0

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction,
                                            ignore_index=self.ignore_index)
        return loss


class LawsClsTrainer(TrainClassification):
    def __init__(self, config):
        super(LawsClsTrainer, self).__init__(config, create_examples=create_examples)
        # self.criterion = FocalLoss()
        self.label_t = []
        self.pre_p =[]
        self.text_train = []
        self.text_dev = []
        self.epoch_num = 0
        self.train_batch_num = 0
        self.eval_batch_num = 0

        self.temp_logits = None
        self.temp_labels = None
        self.all_logits = None
        self.all_labels = None

        self.count = 0
        self.batch_count = 0
        self.auc_count = 0
        self.total_val_loss, self.total_val_acc, self.total_mean_auc, self.total_val_f1 = 0.0, 0.0, 0.0, 0.0
    def cal_loss(self, batch):
        self.logger.debug(batch)
        labels = batch['labels']
        input_data = {'input_ids': batch['input_ids'],
                      'atention_mask': batch['attention_mask'],
                      'token_type_ids': batch['token_type_ids']}
        print(len(batch['input_ids'].squeeze().shape))
        print(batch['input_ids'].squeeze().shape)
        # print(self.tokenizer.decode(batch['input_ids'].squeeze()[0]))   inputs
        # print(''.join(self.tokenizer.decode(batch['input_ids'].squeeze()[0]).split()))


        if len(batch['input_ids'].squeeze().shape) < 2:
            print(batch['input_ids'][0])
            input_data = {'input_ids': batch['input_ids'][0]}
            pred = self.model(input_data)
            # loss = self.criterion(torch.sigmoid(pred.logits), torch.index_select(torch.nonzero(labels[0]), 1, torch.tensor([1]).to(labels.device))[0])
            # label_true = torch.index_select(torch.nonzero(labels[0]), 1, torch.tensor([1]).to(labels.device))[0].detach().cpu().numpy()
            loss = self.criterion(torch.sigmoid(pred.logits), labels[0])
            label_true = torch.index_select(torch.nonzero(labels[0]), 1, torch.tensor([1]).to(labels.device))[0].detach().cpu().numpy()
        else:
            input_data = {'input_ids': batch['input_ids'].squeeze()}
            pred = self.model(input_data)
            # loss = self.criterion(torch.sigmoid(pred.logits),
            #                       torch.index_select(torch.nonzero(labels.squeeze()), 1, torch.tensor([1]).to(labels.device)).squeeze())
            # label_true = torch.index_select(torch.nonzero(labels.squeeze()), 1, torch.tensor([1]).to(labels.device)).squeeze().detach().cpu().numpy()
            temp = torch.sigmoid(pred)
            loss = self.criterion(torch.sigmoid(pred.mean(dim=1)), labels.squeeze())
            label_true = torch.index_select(torch.nonzero(labels.squeeze()), 1, torch.tensor([1]).to(labels.device)).squeeze().detach().cpu().numpy()
        # 计算F1
        # metric = Metric(torch.sigmoid(pred.mean(dim=1)).cpu().detach().numpy(), labels.squeeze().cpu().detach().numpy())
        pred = torch.sigmoid(pred.mean(dim=1))
        # preds = torch.sigmoid(pred) > 0.5
        # pre_max = np.argmax(pred.logits.detach().cpu().numpy(), axis=-1)
        # preds = preds.detach().cpu().numpy().astype(int)

        if self.model.training:
            label_ids = labels.squeeze().float().cuda()
            self.temp_logits = pred if self.temp_logits is None else torch.cat((self.temp_logits, pred), dim=0)
            self.temp_labels = label_ids if self.temp_labels is None else torch.cat((self.temp_labels, label_ids), dim=0)
            # if self.count == 31:
            self.all_logits = self.temp_logits if self.all_logits is None else torch.cat((self.all_logits, self.temp_logits), dim=0)
            self.all_labels = self.temp_labels if self.all_labels is None else torch.cat((self.all_labels, self.temp_labels), dim=0)

            temp_val_loss = self.criterion(self.temp_logits, self.temp_labels).item()
            metric = Metric(self.temp_logits.cpu().detach().numpy(), self.temp_labels.cpu().detach().numpy())
            temp_val_all_acc = metric.accuracy_all()
            temp_val_mean_acc = metric.accuracy_mean()
            temp_val_f1 = metric.fscore_class()
            temp_val_auc = metric.auc()

            if temp_val_auc is not None:
                self.auc_count += 1
                temp_val_mean_auc = temp_val_auc.mean()
                self.total_mean_auc += temp_val_mean_auc
            self.total_val_acc += temp_val_mean_acc
            self.total_val_f1 += temp_val_f1
            self.total_val_loss += temp_val_loss

            self.batch_count += 1
            # 重置
            self.temp_logits = None
            self.temp_labels = None
            # self.count = 0
            self.train_batch_num += 1
            # # for i in range(batch[3].shape[0]):
            # #     R = self.bert_extract_item(outputs[1], outputs[2], i)
            # #     T = self.trueSubject(batch, i)
            # #     self.metric.update(true_subject=T, pred_subject=R)
            # self.label_t = np.append(self.label_t, label_true)
            # self.pre_p = np.append(self.pre_p, pre_max)
            # for i in range(len(batch['input_ids'].squeeze())):
            #     self.text_train.append(''.join(self.tokenizer.decode(batch['input_ids'].squeeze()[i]).split())
            #                            +'-'+str(label_true[i])+'-'+str(pre_max[i]))
            if (self.train_batch_num % (math.ceil(len(self.train_dataset)/self.config["train_batch_size"]))) == 0:
                print('--------')
                print(self.total_val_f1 / self.batch_count, self.total_val_acc / self.batch_count, self.total_val_loss / self.train_batch_num) # , self.total_mean_auc / self.auc_count
                print('--------')
            #     self.epoch_num += 1
            #     # map_df = pd.read_csv("data/fyt_cls_after/bert_loan/loan_all_label.csv")
            #     # print("train_batch_num:" + str(self.train_batch_num))
            #     # print(map_df['label'])
            #     print(classification_report(self.label_t.astype(np.int64), self.pre_p.astype(np.int64)))
                self.temp_logits = None
                self.temp_labels = None
                self.all_logits = None
                self.all_labels = None
                self.count = 0
                self.batch_count = 0
                self.auc_count = 0
                self.total_val_loss, self.total_val_acc, self.total_mean_auc, self.total_val_f1 = 0.0, 0.0, 0.0, 0.0

        else:
            label_ids = labels.squeeze().float().cuda()
            self.temp_logits = pred if self.temp_logits is None else torch.cat((self.temp_logits, pred), dim=0)
            self.temp_labels = label_ids if self.temp_labels is None else torch.cat((self.temp_labels, label_ids), dim=0)
            self.all_logits = self.temp_logits if self.all_logits is None else torch.cat((self.all_logits, self.temp_logits), dim=0)
            self.all_labels = self.temp_labels if self.all_labels is None else torch.cat((self.all_labels, self.temp_labels), dim=0)

            temp_val_loss = self.criterion(self.temp_logits, self.temp_labels).item()
            metric = Metric(self.temp_logits.cpu().detach().numpy(), self.temp_labels.cpu().detach().numpy())
            temp_val_all_acc = metric.accuracy_all()
            temp_val_mean_acc = metric.accuracy_mean()
            temp_val_f1 = metric.fscore_class()
            temp_val_auc = metric.auc()

            if temp_val_auc is not None:
                self.auc_count += 1
                temp_val_mean_auc = temp_val_auc.mean()
                self.total_mean_auc += temp_val_mean_auc
            self.total_val_acc += temp_val_mean_acc
            self.total_val_f1 += temp_val_f1
            self.total_val_loss += temp_val_loss

            self.batch_count += 1
            # 重置
            self.temp_logits = None
            self.temp_labels = None
            # self.count = 0
            self.eval_batch_num += 1
            # self.eval_batch_num += 1
            # self.label_t = np.append(self.label_t, label_true)
            # self.pre_p = np.append(self.pre_p, pre_max)
            # for j in range(len(batch['input_ids'])):
            #     self.text_dev.append(''.join(self.tokenizer.decode(batch['input_ids'].squeeze()[j]).split())
            #                            +'-'+str(label_true[j])+'-'+str(pre_max[j]))
            if (self.eval_batch_num % (math.ceil(len(self.eval_dataset)/self.config["eval_batch_size"]))) == 0: # 一个epoch 输出一次
                print('--------')
                print(self.total_val_f1 / self.batch_count, self.total_val_acc / self.batch_count, self.total_val_loss / self.eval_batch_num) #, self.total_mean_auc / self.auc_count
                print('--------')

            #     # map_df = pd.read_csv("data/fyt_cls_after/bert_loan/loan_all_label.csv")
            #     # print("train_batch_num:" + str(self.eval_batch_num))
            #     print(classification_report(self.label_t.astype(np.int64), self.pre_p.astype(np.int64)))
            #     if self.epoch_num == self.config["num_train_epochs"]:
            #         for item_index_label_t in range(len(self.label_t)):
            #             if int(self.label_t[item_index_label_t]) == 3 and int(self.label_t[item_index_label_t]) != int(self.pre_p[item_index_label_t]):
            #                 print(self.text_dev[item_index_label_t].replace('[PAD]', ''))
            #     self.label_t = []
            #     self.pre_p = []
            #     self.text_dev = []
            #     self.eval_batch_num = 0
                self.temp_logits = None
                self.temp_labels = None
                self.all_logits = None
                self.all_labels = None
                self.count = 0
                self.batch_count = 0
                self.auc_count = 0
                self.total_val_loss, self.total_val_acc, self.total_mean_auc, self.total_val_f1 = 0.0, 0.0, 0.0, 0.0

        # result = []
        # for pred_item in preds:
        #     result.append([index for index, res in enumerate(pred_item) if res == 1])
        # metric = datasets.load_metric('accuracy')

        # loss = self.criterion(torch.sigmoid(pred.logits), labels.squeeze())
        self.logger.debug(loss)
        # loss = (loss - 0.4).abs() + 0.4 # 正则 泛洪法
        return loss


class LawsClsInfer(INferClassification):
    def __init__(self, config):
        super(LawsClsInfer, self).__init__(config, create_examples=create_examples)


if __name__ == '__main__':
    LawsClsTrainer(config="LawEntityExtraction/LabelCls/BertModel/cls_train.yaml").run()
    # LawsClsInfer(config="RelevantLaws/BertModel/infer_config.yaml").run()
