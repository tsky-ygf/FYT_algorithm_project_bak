#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 11:34
# @Author  : Adolf
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
from Tools.train_tool import BaseTrainTool
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertForPreTraining,
)
from BasicTask.SentenceEmbedding.simcse.models import RobertaForCL, BertForCL



class TrainSimces(BaseTrainTool):
    def __init__(self, config_path, model_name):
        # self.bert_config, self.bert_model, self.bert_tokenizer, self.bert_dataset = MODEL_CLASSES[model_name]
        # self.num_labels = len(self.bert_dataset.label_list)
        super(TrainSimces, self).__init__(config_path=config_path)
        self.logger.info(self.config)
        # self.data_collator = ClueNerDataset.data_collator

    def init_model(self):
        tokenizer_kwargs = {
            "cache_dir": self.config.cache_dir,
            "use_fast": self.config.use_fast_tokenizer,
            "revision": self.config.model_revision,
            "use_auth_token": True if self.config.use_auth_token else None,
        }
        if self.config.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, **tokenizer_kwargs)
        elif self.config.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        return tokenizer, model

    # def data_collator(self, batch):
    # return self.bert_dataset.data_collator(batch)

    # def init_dataset(self):
    # train_dataset = self.bert_dataset(data_dir="data/cluener/train.json", tokenizer=self.tokenizer, mode="train",
    #                                   max_length=self.config["max_length"])

    # dev_dataset = self.bert_dataset(data_dir="data/cluener/dev.json", tokenizer=self.tokenizer, mode="dev",
    #                                 max_length=self.config["max_length"])

    # return train_dataset, dev_dataset

    # def cal_loss(self, batch):
    # pass
    # self.logger.info(batch)
    # exit()
    # inputs = {"input_ids": batch[0], "attention_mask": batch[1],
    #           "start_positions": batch[3], "end_positions": batch[4]}
    #
    # outputs = self.model(**inputs)
    # loss = outputs[0]
    # return loss
