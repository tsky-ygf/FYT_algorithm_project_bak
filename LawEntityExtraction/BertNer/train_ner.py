#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/29 09:39
# @Author  : Adolf
# @Site    : 
# @File    : train_ner.py
# @Software: PyCharm
import torch

from Tools.train_tool import BaseTrainTool
from transformers import BertTokenizer, BertConfig
from LawEntityExtraction.BertNer.ModelStructure.bert_ner_model import BertSpanForNer
from LawEntityExtraction.BertNer.model_ner_dataset import ClueNerDataset

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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class TrainLawsNER(BaseTrainTool):
    def __init__(self, config_path):
        super(TrainLawsNER, self).__init__(config_path=config_path)
        self.logger.info(self.config)
        # self.data_collator = ClueNerDataset.data_collator

    def init_model(self):
        tokenizer = BertTokenizer.from_pretrained(self.config["pre_train_tokenizer"], do_lower_case=True)
        bert_config = BertConfig.from_pretrained(self.config["pre_train_model"], num_labels=self.config["num_labels"])
        bert_config.soft_label = self.config["soft_label"]
        bert_config.loss_type = self.config["loss_type"]
        model = BertSpanForNer.from_pretrained(self.config["pre_train_model"], config=bert_config)
        # model.zero_grad()
        return tokenizer, model

    def data_collator(self, batch):
        return ClueNerDataset.data_collator(batch)

    def init_dataset(self):
        train_dataset = ClueNerDataset(data_dir="data/cluener/train.json", tokenizer=self.tokenizer, mode="train",
                                       max_length=self.config["max_length"])

        dev_dataset = ClueNerDataset(data_dir="data/cluener/dev.json", tokenizer=self.tokenizer, mode="dev",
                                     max_length=self.config["max_length"])

        return train_dataset, dev_dataset

    def cal_loss(self, batch):
        # pass
        # self.logger.info(batch)
        # exit()
        inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                  "start_positions": batch[3], "end_positions": batch[4]}

        outputs = self.model.forward(**inputs)
        loss = outputs[0]
        return loss

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
        # args.warmup_steps = int(t_total * args.warmup_proportion)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config["learning_rate"], eps=1e-08)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        #                                             num_training_steps=t_total)
        return optimizer


if __name__ == '__main__':
    TrainLawsNER(config_path="LawEntityExtraction/BertNer/Config/base_ner_config.yaml").train_main()
    # TrainLawsCls(config_path="RelevantLaws/Config/base_laws_cls_model.yaml").eval_epoch()
