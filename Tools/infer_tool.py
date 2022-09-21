#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 13:34
# @Author  : Adolf
# @Site    : 
# @File    : infer_tool.py
# @Software: PyCharm
import torch
from tqdm.auto import tqdm

from Utils.logger import get_module_logger
from Tools.parse_argument import parse_config_file
from torch.utils import data

from transformers import default_data_collator
from Tools.data_pipeline import BaseDataset
from pprint import pformat


class BaseInferTool:
    def __init__(self, config, create_examples):
        self.config = parse_config_file(config)
        self.logger = get_module_logger(module_name="Infer", level=self.config.get("log_level", "INFO"))
        if "device" in self.config:
            self.device = torch.device(self.config["device"])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(pformat(self.config))
        self.tokenizer, self.model = self.init_model()
        self.model.eval()
        self.model.to(self.device)

        self.create_examples = create_examples

        self.test_dataset = self.init_dataset()
        self.test_dataloader = self.init_dataloader()

        # self.process_bar = tqdm(range(len(self.test_dataset)), desc="Infer")

    def init_model(self):
        raise NotImplementedError

    def init_dataset(self, *args, **kwargs):
        data_dir_dict = {'test': self.config['test_data_path']}

        test_dataset = BaseDataset(data_dir_dict,
                                   tokenizer=self.tokenizer,
                                   mode='test',
                                   max_length=self.config['max_len'],
                                   create_examples=self.create_examples,
                                   is_debug=self.config['is_debug'])
        return test_dataset

    def data_collator(self, *args, **kwargs):
        self.logger.info("Use default data collator")
        return default_data_collator(*args, **kwargs)

    def init_dataloader(self):
        test_dataloader = data.DataLoader(
            dataset=self.test_dataset, shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.config["test_batch_size"],
        )
        return test_dataloader

    def metrics(self, y_true, y_pred):
        raise NotImplementedError

    def infer(self, text):
        raise NotImplementedError

    def run(self):
        return_result = []
        return_labels = []

        for batch in tqdm(self.test_dataloader):
            res, label = self.cal_res(batch)
            if res is not None and label is not None:
                return_result.append(res)
                return_labels.append(label)

        self.metrics(return_labels, return_result)

    def cal_res(self, batch):
        raise NotImplementedError
