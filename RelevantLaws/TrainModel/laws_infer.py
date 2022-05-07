#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 11:24
# @Author  : Adolf
# @Site    : 
# @File    : laws_infer.py
# @Software: PyCharm
import os
import torch
import pandas as pd

from tqdm.auto import tqdm
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import MultiLabelBinarizer

from Utils.logger import get_module_logger
from Utils.parse_file import parse_config_file

from RelevantLaws.DataProcess.laws_model_dataset import prepare_input, LawsThuTestDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RelebantLawsInferTool:
    def __init__(self, config_file):
        self.config = parse_config_file(config_file)
        self.logger = get_module_logger(module_name="Infer", level=self.config.get("log_level", "INFO"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(self.config)

        self.tokenizer, self.model = self.init_model()
        map_df = pd.read_csv(self.config["label_mapping_path"])
        self.pred_map = dict(zip(map_df['index'], map_df['laws']))
        self.label_map = dict(zip(map_df['laws'], map_df['index']))

        # self.logger.debug(self.pred_map)

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["infer_tokenizer"])
        model = AutoModelForSequenceClassification.from_pretrained(self.config["infer_model"])
        model.to(self.device)
        return tokenizer, model

    def infer(self, text):
        inputs = prepare_input(text, self.tokenizer, max_len=self.config["max_len"])
        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits) > 0.5
        predictions = predictions.detach().numpy().astype(int)
        self.logger.info(predictions[0])
        result = [self.pred_map[index] for index, res in enumerate(predictions[0]) if res == 1]
        self.logger.info(result)

    @staticmethod
    def collate_fn(batch):
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

    # @staticmethod
    # def f1_sampled(actual, pred):
    #     # converting the multi-label classification to a binary output
    #     mlb = MultiLabelBinarizer()
    #     actual = mlb.fit_transform(actual)
    #     pred = mlb.fit_transform(pred)
    #
    #     # fitting the data for calculating the f1 score
    #     f1 = f1_score(actual, pred, average="samples")
    #     return f1

    def cal_accuracy(self, y_true, y_pred):
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

        f1_score = f1_score / len(y_true)
        recall_score = recall_score / len(y_true)
        precision_score = precision_score / len(y_true)

        self.logger.info(
            '模型测试集上的测试结果为: recall:{},precision:{},f1-score为:{}'.format(recall_score, precision_score, f1_score))
        return f1_score

    def test_data(self):
        test_dataset = LawsThuTestDataset(self.tokenizer, self.config, self.label_map)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config["batch_size"],
                                                      collate_fn=self.collate_fn)

        return_result = []
        return_labels = []

        for batch in tqdm(test_dataloader):
            input_data, labels = batch
            input_data = {k: v.to(self.device) for k, v in input_data.items()}
            out = self.model(**input_data)
            preds = torch.sigmoid(out.logits) > 0.5
            preds = preds.detach().cpu().numpy().astype(int)

            result = []
            for pred in preds:
                result.append([index for index, res in enumerate(pred) if res == 1])

            # self.logger.debug(result)
            # self.logger.debug(labels)

            for res, label in zip(result, labels):
                if len(res) > 0 and len(label) > 0:
                    return_labels.append(label)
                    return_result.append(res)

        self.cal_accuracy(return_labels, return_result)


if __name__ == '__main__':
    infer_tool = RelebantLawsInferTool("RelevantLaws/Config/base_laws_cls_infer.yaml")
    # infer_tool.infer(text="在当事人之间产生的特定的权利和义务关系，享有权利的人是债权人，负有义务的人是债务人。")
    infer_tool.test_data()
