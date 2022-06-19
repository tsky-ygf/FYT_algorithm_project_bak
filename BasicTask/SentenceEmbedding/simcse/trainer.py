#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 11:34
# @Author  : Adolf
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
# import traceback

# try:
from regex import F
from Tools.train_tool import BaseTrainTool

from transformers import AutoTokenizer,BertForPreTraining
from BasicTask.SentenceEmbedding.simcse.models import RobertaForCL, BertForCL



class TrainSimces(BaseTrainTool):
    def __init__(self, config_path):
        # self.bert_config, self.bert_model, self.bert_tokenizer, self.bert_dataset = MODEL_CLASSES[model_name]
        # self.num_labels = len(self.bert_dataset.label_list)
        super(TrainSimces, self).__init__(config_path=config_path)
        self.logger.info(self.config)
        # self.data_collator = ClueNerDataset.data_collator

    def init_model(self):
        tokenizer_kwargs = {
            "cache_dir": self.config['tokenizer_cache_dir'],
            "use_fast": self.config['use_fast_tokenizer'],
            "revision": self.config['model_revision'],
            "use_auth_token": True if self.config['use_auth_token'] else None,
        }

        if 'pre_train_tokenizer' in self.config:
            tokenizer = AutoTokenizer.from_pretrained(self.config['pre_train_tokenizer'], **tokenizer_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config['pre_train_model'], **tokenizer_kwargs)

        if 'pre_train_model' in self.config:
            if 'roberta' in self.config['pre_train_model']:
                model = RobertaForCL.from_pretrained(
                    self.config['pre_train_model'],
                    from_tf=False,
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=self.config['model_revision'],
                    use_auth_token=True if self.config['use_auth_token'] else None,
                    model_args=model_args                  
                )
            elif 'bert' in self.config['pre_train_model']:
                model = BertForCL.from_pretrained(
                    self.config['pre_train_model'],
                    from_tf=False,
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=self.config['model_revision'],
                    use_auth_token=True if self.config['use_auth_token'] else None,
                    model_args=model_args
                )
                if model_args.do_mlm:
                    pretrained_model = BertForPreTraining.from_pretrained(self.config['pre_train_model'])
                    model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())

        model.resize_token_embeddings(len(tokenizer))
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

if __name__ == '__main__':
    TrainSimces(config_path="BasicTask/SentenceEmbedding/simcse/config.yaml").train_main()

