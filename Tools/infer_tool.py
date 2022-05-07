#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 13:34
# @Author  : Adolf
# @Site    : 
# @File    : infer_tool.py
# @Software: PyCharm
import os
import torch
from tqdm.auto import tqdm

from Utils.logger import get_module_logger
from Utils.parse_file import parse_config_file
from torch.utils import data

from transformers import default_data_collator


class BaseInferTool:
    def __init__(self, config_file):
        self.config = parse_config_file(config_file)
        self.logger = get_module_logger(module_name="Infer", level=self.config.get("log_level", "INFO"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(self.config)

        self.tokenizer, self.model = self.init_model()
        self.model.eval()

        self.test_dataset = self.init_dataset()
        self.test_dataloader = self.init_dataloader()

    def init_model(self):
        raise NotImplementedError

    def init_dataset(self, *args, **kwargs):
        raise NotImplemented

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

    def infer(self, text):
        raise NotImplementedError

    def test_data(self):
        raise NotImplementedError
