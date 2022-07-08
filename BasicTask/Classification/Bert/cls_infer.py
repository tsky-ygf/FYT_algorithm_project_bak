#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/8 16:27
# @Author  : Adolf
# @Site    : 
# @File    : cls_infer.py
# @Software: PyCharm
import os

import torch
import pandas as pd

from Tools.infer_tool import BaseInferTool
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class INferClassification(BaseInferTool):
    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["infer_tokenizer"])
        model = AutoModelForSequenceClassification.from_pretrained(self.config["infer_model"])
        model.to(self.device)
        return tokenizer, model

    def metrics(self, y_true, y_pred):
        f1_score = 0
        recall_score = 0
        precision_score = 0

        for one_true, one_pred in zip(y_true, y_pred):
            # print(one_true, one_pred)
            # print(one_f1)
            recall = len(set(one_true).intersection(set(one_pred))) / len(one_true)
            precision = len(set(one_true).intersection(set(one_pred))) / len(one_pred)
            recall_score += recall
            precision_score += precision

            try:
                f1 = 2 * recall * precision / (recall + precision)
            except Exception as e:
                self.logger.debug(e)
                # self.logger.debug()
                f1 = 0
            f1_score += f1

        f1_score /= len(y_true)
        recall_score /= len(y_true)
        precision_score /= len(y_true)

        self.logger.info(
            '模型测试集上的测试结果为: recall:{},precision:{},f1-score为:{}'.format(recall_score, precision_score, f1_score))
        return f1_score

    def cal_res(self, batch):
        labels = batch['labels']
        input_data = {'input_ids': batch['input_ids'], }
        # 'atention_mask': batch['attention_mask'],
        # 'token_type_ids': batch['token_type_ids']}
        input_data = {k: v.to(self.device) for k, v in input_data.items()}
        out = self.model(input_data['input_ids'].squeeze())
        preds = torch.sigmoid(out.logits) > 0.5
        preds = preds.detach().cpu().numpy().astype(int)

        result = []
        for pred in preds:
            result.append([index for index, res in enumerate(pred) if res == 1])

        for res, label in zip(result, labels):
            if len(res) > 0 and len(label) > 0:
                return res, label

    def infer(self, text):
        map_df = pd.read_csv(self.config["label_mapping_path"])
        pred_map = dict(zip(map_df['index'], map_df['laws']))
        # label_map = dict(zip(map_df['laws'], map_df['index']))

        # inputs = prepare_input(text, self.tokenizer, max_len=self.config["max_len"])
        inputs = self.tokenizer(text,
                                add_special_tokens=True,
                                max_length=self.config['max_len'],
                                padding="max_length",
                                truncation=True,
                                return_offsets_mapping=False,
                                return_tensors="pt")

        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits) > 0.5
        predictions = predictions.detach().numpy().astype(int)
        self.logger.info(predictions[0])
        result = [pred_map[index] for index, res in enumerate(predictions[0]) if res == 1]
        self.logger.info(result)
