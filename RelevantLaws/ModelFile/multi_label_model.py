#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 23:15
# @Author  : Adolf
# @Site    : 
# @File    : multi_label_model.py
# @Software: PyCharm
import torch
from torch import nn
from transformers import AutoModel


# Multi-label classification model
class MultiLabelClsModel(nn.Module):
    def __init__(self, config):
        super(MultiLabelClsModel, self).__init__()
        self.config = config

        self.model = AutoModel.from_pretrained(self.config["pre_train_model_path"])
        hidden_size = self.model.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(self.config["dropout"])
        # self.fc1 = nn.Linear(hidden_size, self.config["feature_dim"])
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(self.config["dropout"])
        # self.fc2 = nn.Linear(self.config["feature_dim"], self.config["num_labels"])

        self.dense = nn.Sequential(
            nn.Linear(hidden_size, self.config["feature_dim"]),
            nn.SiLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(self.config["feature_dim"], self.config["num_labels"])
        )

        # self._init_weights(self.fc)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs['last_hidden_state']
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.layer_norm(feature)
        output = self.dropout(output)
        output = self.dense(output)
        # output = self.fc1(output)
        # output = self.relu(output)
        # output = self.fc2(output)
        return output
