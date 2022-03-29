#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 13:34
# @Author  : Adolf
# @Site    : 
# @File    : infer_tool.py
# @Software: PyCharm
import torch
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from Utils.logger import get_module_logger
from Utils.parse_file import parse_config_file

from RelevantLaws.DataProcess.laws_model_dataset import prepare_input, LawsThuTestDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BaseInferTool:
    def __init__(self, config_file):
        self.config = parse_config_file(config_file)
        self.logger = get_module_logger(module_name="Infer", level=self.config.get("log_level", "INFO"))

        self.logger.debug(self.config)

        self.tokenizer, self.model = self.init_model()
        map_df = pd.read_csv(self.config["label_mapping_path"])
        self.pred_map = dict(zip(map_df['index'], map_df['laws']))
        self.label_map = dict(zip(map_df['laws'], map_df['index']))

        self.logger.debug(self.pred_map)

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
        result = [self.pred_map[index] for index, res in enumerate(predictions[0]) if res == 1]
        self.logger.info(result)

    def collate_fn(self, batch):
        self.logger.debug(batch)

        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        labels_list = []
        for one_data in batch:
            input_ids_list.append(one_data['input_ids'])
            token_type_ids_list.append(one_data['token_type_ids'])
            attention_mask_list.append(one_data['attention_mask'])
            labels_list.append(one_data['labels'])

        input_ids = torch.stack(input_ids_list, dim=0)
        token_type_ids = torch.stack(token_type_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)

        input_ids = torch.squeeze(input_ids)
        token_type_ids = torch.squeeze(token_type_ids)
        attention_mask = torch.squeeze(attention_mask)

        input_data = {"input_ids": input_ids,
                      "token_type_ids": token_type_ids,
                      "attention_mask": attention_mask, }

        return input_data, labels_list


    def f1_sampled(actual, pred):
        # converting the multi-label classification to a binary output
        mlb = MultiLabelBinarizer()
        actual = mlb.fit_transform(actual)
        pred = mlb.fit_transform(pred)

        # fitting the data for calculating the f1 score
        f1 = f1_score(actual, pred, average="samples")
        return f1

    def test_data(self):
        test_dataset = LawsThuTestDataset(self.tokenizer, self.config, self.label_map)
        item_ = test_dataset[10]
        self.logger.debug(item_)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config["batch_size"],
                                                      collate_fn=self.collate_fn)

        for batch in test_dataloader:
            input_data, labels = batch
            out = self.model(**input_data)
            preds = torch.sigmoid(out.logits) > 0.5
            preds = preds.detach().numpy().astype(int)

            result = []
            for pred in preds:
                result.append([index for index, res in enumerate(pred) if res == 1])

            self.logger.debug(result)
            self.logger.debug(labels)

            break


if __name__ == '__main__':
    infer_tool = BaseInferTool("RelevantLaws/Config/base_laws_cls_infer.yaml")
    # infer_tool.infer(text="在当事人之间产生的特定的权利和义务关系，享有权利的人是债权人，负有义务的人是债务人。")
    infer_tool.test_data()
