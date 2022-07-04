#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 11:34
# @Author  : Adolf
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
# import traceback
from Tools.train_tool import BaseTrainTool

from transformers import AutoTokenizer, BertForPreTraining, AutoConfig
from BasicTask.SentenceEmbedding.simcse.models import RobertaForCL, BertForCL

from datasets import load_dataset


class TrainSimces(BaseTrainTool):
    def __init__(self, config_path):
        # self.bert_config, self.bert_model, self.bert_tokenizer, self.bert_dataset = MODEL_CLASSES[model_name]
        # self.num_labels = len(self.bert_dataset.label_list)
        super(TrainSimces, self).__init__(config_path=config_path)
        self.logger.info(self.config)
        # self.data_collator = ClueNerDataset.data_collator

    def init_model(self):
        config_kwargs = {
            "cache_dir": self.config['cache_dir'],
            "revision": self.config['model_revision'],
            "use_auth_token": True if self.config['use_auth_token'] else None,
        }
        if 'config_name' in self.config:
            config = AutoConfig.from_pretrained(self.config['config_name'], **config_kwargs)
        elif 'pre_train_model' in self.config:
            config = AutoConfig.from_pretrained(self.config['pre_train_model'], **config_kwargs)
        else:
            self.logger.warning("You are instantiating a new config instance from scratch.")
            raise ValueError('Not Implemented Another Model')

        tokenizer_kwargs = {
            "cache_dir": self.config['cache_dir'],
            "use_fast": self.config['use_fast_tokenizer'],
            "revision": self.config['model_revision'],
            "use_auth_token": True if self.config['use_auth_token'] else None,
        }

        if 'pre_train_tokenizer' in self.config:
            tokenizer = AutoTokenizer.from_pretrained(self.config['pre_train_tokenizer'], **tokenizer_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.config['pre_train_model'], **tokenizer_kwargs)

        if 'roberta' in self.config['pre_train_model']:
            model = RobertaForCL.from_pretrained(
                self.config['pre_train_model'],
                from_tf=False,
                config=config,
                cache_dir=self.config['cache_dir'],
                revision=self.config['model_revision'],
                use_auth_token=True if self.config['use_auth_token'] else None,
                model_kargs=self.config,
            )
        elif 'bert' in self.config['pre_train_model']:
            model = BertForCL.from_pretrained(
                self.config['pre_train_model'],
                from_tf=False,
                config=config,
                cache_dir=self.config['cache_dir'],
                revision=self.config['model_revision'],
                use_auth_token=True if self.config['use_auth_token'] else None,
                model_kargs=self.config,
            )
            if self.config['do_mlm']:
                pretrained_model = BertForPreTraining.from_pretrained(self.config['pre_train_model'])
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise ValueError('Not Implemented Another Model')

        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    # def data_collator(self, batch):
    # return self.bert_dataset.data_collator(batch)

    def init_dataset(self):
        datasets = load_dataset("text", data_files={
            'train': '/home/fyt/code_hup/third_part_code/SimCSE/data/wiki1m_for_simcse.txt'},
                                cache_dir="/home/fyt/code_hup/third_part_code/SimCSE/data/")

        column_names = datasets["train"].column_names
        self.sent2_cname = None
        if len(column_names) == 2:
            # Pair datasets
            self.sent0_cname = column_names[0]
            self.sent1_cname = column_names[1]
        elif len(column_names) == 3:
            # Pair datasets with hard negatives
            self.sent0_cname = column_names[0]
            self.sent1_cname = column_names[1]
            self.sent2_cname = column_names[2]
        elif len(column_names) == 1:
            # Unsupervised datasets
            self.sent0_cname = column_names[0]
            self.sent1_cname = column_names[0]
        else:
            raise NotImplementedError

        # if training_args.do_train:
        train_dataset = datasets["train"].map(
            self.prepare_features,
            batched=True,
            num_proc=self.config['preprocessing_num_workers'],
            remove_columns=column_names,
            load_from_cache_file=not self.config['overwrite_cache'],
        )

        return train_dataset, train_dataset

    def prepare_features(self, examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[self.sent0_cname])

        # Avoid "None" fields
        for idx in range(total):
            if examples[self.sent0_cname][idx] is None:
                examples[self.sent0_cname][idx] = " "
            if examples[self.sent1_cname][idx] is None:
                examples[self.sent1_cname][idx] = " "

        sentences = examples[self.sent0_cname] + examples[self.sent1_cname]

        # If hard negative exists
        if self.sent2_cname is not None:
            for idx in range(total):
                if examples[self.sent2_cname][idx] is None:
                    examples[self.sent2_cname][idx] = " "
            sentences += examples[self.sent2_cname]

        sent_features = self.tokenizer(
            sentences,
            max_length=self.config['max_length'],
            truncation=True,
            padding="max_length",
        )

        features = {}
        if self.sent2_cname is not None:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]] for i in
                    range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]

        return features

    def cal_loss(self, batch):
        self.logger.debug(batch)
        # exit()
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss


if __name__ == '__main__':
    TrainSimces(config_path="BasicTask/SentenceEmbedding/simcse/config.yaml").train_main()
