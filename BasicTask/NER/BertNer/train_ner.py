#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 09:39
# @Author  : Adolf
# @Site    : 
# @File    : train_ner.py
# @Software: PyCharm
import math

from Tools.train_tool import BaseTrainTool
from transformers import BertTokenizer, BertConfig
from BasicTask.NER.BertNer.ModelStructure.bert_ner_model import BertSpanForNer, BertCrfForNer
from BasicTask.NER.BertNer.model_ner_dataset import ClueNerSpanDataset, ClueNerCRFDataset, LoanNerSpanDataset

from metrics import SpanEntityScore


import os
import random
import torch
import numpy as np

# import random
# import os
# import numpy as np
# seed = 42
#
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# some cudnn methods can be random even after fixing the seed
# unless you tell it to be deterministic
# torch.backends.cudnn.deterministic = True

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    "bert_crf": (BertConfig, BertCrfForNer, BertTokenizer, ClueNerCRFDataset),
    "bert_span": (BertConfig, BertSpanForNer, BertTokenizer, ClueNerSpanDataset),
    "bert_span_loan": (BertConfig, BertSpanForNer, BertTokenizer, LoanNerSpanDataset),
}


class TrainLawsNER(BaseTrainTool):
    def __init__(self, config_path, model_name):
        self.bert_config, self.bert_model, self.bert_tokenizer, self.bert_dataset = MODEL_CLASSES[model_name]
        self.num_labels = len(self.bert_dataset.label_list)
        super(TrainLawsNER, self).__init__(config_path=config_path)
        self.logger.info(self.config)
        # self.data_collator = ClueNerDataset.data_collator
        self.metric = SpanEntityScore()
        self.eval_batch_num = 0
        self.train_batch_num = 0

    def init_model(self):
        tokenizer = self.bert_tokenizer.from_pretrained(self.config["pre_train_tokenizer"], do_lower_case=True)
        bert_model_config = self.bert_config.from_pretrained(self.config["pre_train_model"], num_labels=self.num_labels)
        bert_model_config.soft_label = self.config["soft_label"]
        bert_model_config.loss_type = self.config["loss_type"]
        model = self.bert_model.from_pretrained(self.config["pre_train_model"], config=bert_model_config)
        # model.zero_grad()
        return tokenizer, model

    def data_collator(self, batch):
        return self.bert_dataset.data_collator(batch)

    def init_dataset(self):
        train_dataset = self.bert_dataset(data_dir=self.config["train_data_dir"], dataset_name=self.config["dataset_name"],  tokenizer=self.tokenizer, mode="train",
                                          max_length=self.config["max_length"])

        dev_dataset = self.bert_dataset(data_dir=self.config["dev_data_dir"], dataset_name=self.config["dataset_name"], tokenizer=self.tokenizer, mode="dev",
                                        max_length=self.config["max_length"])

        return train_dataset, dev_dataset

    def cal_loss(self, batch):
        # pass
        # self.logger.info(batch)
        # exit()
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                  "start_positions": batch[3], "end_positions": batch[4]}

        outputs = self.model(**inputs)

        weight = self.model.start_fc.dense.weight[::,::]

        loss = outputs[0]
        # 记录结果
        if self.model.training:
            self.train_batch_num += 1
            for i in range(batch[3].shape[0]):
                R = self.bert_extract_item(outputs[1], outputs[2], i)
                T = self.trueSubject(batch, i)
                self.metric.update(true_subject=T, pred_subject=R)
            if (self.train_batch_num % (math.ceil(len(self.train_dataset)/self.config["train_batch_size"]))) == 0:
                print(weight)
                print("train_batch_num:" + str(self.train_batch_num))
                print(self.metric.result())
                self.metric.reset()
                self.train_batch_num = 0
        if not self.model.training:
            self.eval_batch_num += 1
            for i in range(batch[3].shape[0]):
                R = self.bert_extract_item(outputs[1], outputs[2], i)
                T = self.trueSubject(batch, i)
                self.metric.update(true_subject=T, pred_subject=R)
            if (self.eval_batch_num % (math.ceil(len(self.eval_dataset)/self.config["eval_batch_size"]))) == 0:
                print("eval_batch_num:" + str(self.eval_batch_num))
                print(self.metric.result())
                self.metric.reset()
                self.eval_batch_num = 0
            # if 69 * 0 < self.eval_batch_num and self.eval_batch_num < 69 * 1:  # 第八个eval epoch 开始记录metric
            #     for i in range(batch[3].shape[0]):
            #         R = self.bert_extract_item(outputs[1], outputs[2], i)
            #         T = self.trueSubject(batch, i)
            #         self.metric.update(true_subject=T, pred_subject=R)




        return loss

    def bert_extract_item(self, start_logits, end_logits, p):
        S = []
        start_pred = torch.argmax(start_logits, -1).cpu().numpy()[p][1:-1]
        end_pred = torch.argmax(end_logits, -1).cpu().numpy()[p][1:-1]
        for i, s_l in enumerate(start_pred):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_pred[i:]):
                if s_l == e_l:
                    S.append((s_l, i, i + j))
                    break
        return S

    def trueSubject(self, batch, i):
        T = []
        label = []
        start_index, end_index = 0, 0
        item_start = batch[3][i].cpu().numpy()
        item_end = batch[4][i].cpu().numpy()
        for index_i in range(len(item_start)):
            if item_start[index_i] != 0:
                label.append(item_start[index_i])
        for label_i in range(len(label)):
            for index_s in range(len(item_start)):
                if item_start[index_s] != 0 and item_start[index_s] == label[label_i]:
                    start_index = index_s - 1
            for index_e in range(len(item_end)):
                if item_end[index_e] != 0 and item_end[index_e] == label[label_i]:
                    end_index = index_e - 1
            T.append((label[label_i], start_index, end_index))

        # for index_i in range(len(item_sum)):
        #     if item_sum[index_i] != 0:
        #         num += 1
        #         type_sec = item_sum[index_i]
        #         if num % 2 != 0:
        #             start_index = index_i - 1
        #         else:
        #             end_index = index_i - 1
        #             T.append((type_sec, start_index, end_index))
        return T

class TrainSpanForNerClue(TrainLawsNER):
    def __init__(self, config_path):
        model_name = "bert_span"
        super(TrainSpanForNerClue, self).__init__(config_path=config_path, model_name=model_name)
        # self.seed_everything()
    def init_optimizer(self):
        self.logger.info("init optimizer of bert span for ner")
        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = self.model.bert.named_parameters()
        start_parameters = self.model.start_fc.named_parameters()
        end_parameters = self.model.end_fc.named_parameters()
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config["weight_decay"], 'lr': self.config["learning_rate"]},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.config["learning_rate"]},

            {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config["weight_decay"], 'lr': self.config["start_learning_rate"]},
            {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.config["start_learning_rate"]},

            {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config["weight_decay"], 'lr': self.config["end_learning_rate"]},
            {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.config["end_learning_rate"]},
        ]

        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": self.config["weight_decay"], 'lr': self.config["learning_rate"]},
        #     {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0, 'lr': self.config["learning_rate"]},
        #
        #     {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": self.config["weight_decay"], 'lr': 0.001},
        #     {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001},
        #
        #     {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": self.config["weight_decay"], 'lr': 0.001},
        #     {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001},
        # ]

        # args.warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"], eps=1e-08)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        #                                             num_training_steps=t_total)
        return optimizer

    def seed_everything(self, seed=42):
        '''
        设置整个开发环境的seed
        :param seed:
        :param device:
        :return:
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

class TrainSpanForNerLoan(TrainLawsNER):
    def __init__(self, config_path):
        model_name = "bert_span_loan"
        super(TrainSpanForNerLoan, self).__init__(config_path=config_path, model_name=model_name)
        # self.seed_everything()
    def init_optimizer(self):
        self.logger.info("init optimizer of bert span for ner")
        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = self.model.bert.named_parameters()
        start_parameters = self.model.start_fc.named_parameters()
        end_parameters = self.model.end_fc.named_parameters()
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config["weight_decay"], 'lr': self.config["learning_rate"]},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.config["learning_rate"]},

            {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config["weight_decay"], 'lr': self.config["start_learning_rate"]},
            {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.config["start_learning_rate"]},

            {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config["weight_decay"], 'lr': self.config["end_learning_rate"]},
            {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                , 'lr': self.config["end_learning_rate"]},
        ]

        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": self.config["weight_decay"], 'lr': self.config["learning_rate"]},
        #     {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0, 'lr': self.config["learning_rate"]},
        #
        #     {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": self.config["weight_decay"], 'lr': 0.001},
        #     {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001},
        #
        #     {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": self.config["weight_decay"], 'lr': 0.001},
        #     {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        #         , 'lr': 0.001},
        # ]

        # args.warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"], eps=1e-08)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        #                                             num_training_steps=t_total)
        return optimizer

    def seed_everything(self, seed=42):
        '''
        设置整个开发环境的seed
        :param seed:
        :param device:
        :return:
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

class TrainCrfForNerClue(TrainLawsNER):
    def __init__(self, config_path):
        model_name = "bert_crf"
        super(TrainCrfForNerClue, self).__init__(config_path=config_path, model_name=model_name)
        # self.num_labels = self.config["num_labels"]

    def init_optimizer(self):
        self.logger.info("init optimizer of bert crf for ner")
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(self.model.bert.named_parameters())
        crf_param_optimizer = list(self.model.crf.named_parameters())
        linear_param_optimizer = list(self.model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config["weight_decay"], 'lr': self.config["learning_rate"]},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.config["learning_rate"]},

            {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config["weight_decay"], 'lr': self.config["crf_learning_rate"]},
            {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.config["crf_learning_rate"]},

            {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config["weight_decay"], 'lr': self.config["crf_learning_rate"]},
            {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.config["crf_learning_rate"]}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"],
                                      eps=1e-08)
        return optimizer

    def cal_loss(self, batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        outputs = self.model(**inputs)
        loss = outputs[0]
        return loss


if __name__ == '__main__':
    TraSpan = TrainSpanForNerLoan(config_path="BasicTask/NER/BertNer/Config_bak/base_ner_config.yaml")
    TraSpan.train_main()
    print(TraSpan.metric.result())
    # TrainSpanForNer(config_path="huangyulin/project/fyt/LawEntityExtraction/BertNer/Config_bak/base_ner_config.yaml").train_main()


    # TrainCrfForNer(config_path="huangyulin/project/fyt/LawEntityExtraction/BertNer/Config_bak/base_ner_config.yaml").train_main()