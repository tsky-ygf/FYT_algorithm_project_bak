#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 13:34
# @Author  : Adolf
# @Site    : 
# @File    : infer_tool.py
# @Software: PyCharm
import torch

from Utils.logger import get_module_logger
from Utils.parse_file import parse_config_file

from RelevantLaws.DataProcess.laws_model_dataset import prepare_input
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BaseInferTool:
    def __init__(self, config_file):
        self.config = parse_config_file(config_file)
        self.logger = get_module_logger(module_name="Infer", level=self.config.get("log_level", "INFO"))

        self.logger.debug(self.config)

        self.tokenizer, self.model = self.init_model()

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["infer_tokenizer"])
        model = AutoModelForSequenceClassification.from_pretrained(self.config["infer_model"])
        return tokenizer, model

    def infer(self, text):
        inputs = prepare_input(text, self.tokenizer, max_len=self.config["max_len"])
        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits) > 0.5
        predictions = predictions.detach().numpy().astype(int)
        self.logger.info(predictions[0])

if __name__ == '__main__':
    infer_tool = BaseInferTool("RelevantLaws/Config/base_laws_cls_infer.yaml")
    infer_tool.infer(text="在当事人之间产生的特定的权利和义务关系，享有权利的人是债权人，负有义务的人是债务人。")