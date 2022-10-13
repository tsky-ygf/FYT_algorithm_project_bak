#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 13:57
# @Author  : Adolf
# @Site    : 
# @File    : simcse_optim.py
# @Software: PyCharm
import random

import os
import pandas as pd

from data_aug import word_repetition, random_reverse_order
from IntelligentConsultation.src.faq_pipeline.core_model.common_model import MODEL_REGISTRY, SimCSE
from IntelligentConsultation.src.faq_pipeline.main import faq_main
from IntelligentConsultation.src.faq_pipeline.core_model.st_losses import MNRRDropLoss

# from paddlenlp.dataaug import WordSubstitute

from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(290)


# aug = WordSubstitute('synonym', create_n=1, aug_percent=0.3)


@MODEL_REGISTRY.register()
class SimCSE_RDRop(SimCSE):

    def init_data(self):
        train_data = []
        train_df = pd.read_csv(self.config.train_data_path)
        for index, row in train_df.iterrows():
            query1 = row["question"]
            query2 = row["question"]

            # TODO 使用同义词进行数据增强
            # paddlenlp的同义词替换，降低了准确率5个点。

            # if random.random() > 0.8:
            #     query1 = random_reverse_order(query1)
            #     query2 = random_reverse_order(query2)

            if random.random() < 0.2:
                query1 = word_repetition(query1)
                query2 = word_repetition(query2)

            train_data.append(InputExample(texts=[query1, query2]))

        return train_data

    def init_model(self):
        word_embedding_model = models.Transformer(self.config.model_name, max_seq_length=self.config.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        loss = losses.MultipleNegativesRankingLoss(model)
        # loss = MNRRDropLoss(model, kl_weight=0.1)
        return model, loss


if __name__ == '__main__':
    use_config = {
        "model_type": "SimCSE_RDRop",
        "index_name": "topic_qa_test_v2",
        "train_config": {
            "log_level": "INFO",
            "model_name": "model/language_model/chinese-roberta-wwm-ext",
            "model_output_path": "model/similarity_model/simcse-model-optim",
            "train_data_path": "data/fyt_train_use_data/QA/pro_qa.csv",
            "lr": 1e-5,
            "train_batch_size": 128,
            "max_seq_length": 64,
            "num_epochs": 1}
    }

    acc, res_df = faq_main(config=use_config)
    res_df.to_csv("data/fyt_train_use_data/QA/pro_qa_res.csv", index=False)
    print(acc)
