#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/9 15:25
# @Author  : Adolf
# @Site    : 
# @File    : model.py
# @Software: PyCharm
import torch
from torch import nn, Tensor
from sentence_transformers import models, util
from typing import Iterable, Dict, List


# from sentence_transformers import SentenceTransformer


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=util.cos_sim):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # def forward(self, sentence_features: Iterable[Dict[str, Tensor]]):
    def forward(self, reps):
        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}


class SimCSE(nn.Module):
    def __init__(self, model_name, max_seq_length):
        super(SimCSE, self).__init__()

        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        # self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.model = nn.Sequential(word_embedding_model, pooling_model)
        self.loss = MultipleNegativesRankingLoss()

    def forward(self, input_ids_a: Tensor, attention_mask_a: Tensor, input_ids_b: Tensor, attention_mask_b: Tensor):
        feature_a = {"input_ids": input_ids_a, "attention_mask": attention_mask_a}
        feature_b = {"input_ids": input_ids_b, "attention_mask": attention_mask_b}
        sentence_features = [feature_a, feature_b]
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        train_loss = self.loss(reps)

        return reps, train_loss


if __name__ == '__main__':
    simcse_model = SimCSE('model/language_model/bert-base-chinese', 128)

    r_, l_ = simcse_model()
    print(r_)
    print(l_)
