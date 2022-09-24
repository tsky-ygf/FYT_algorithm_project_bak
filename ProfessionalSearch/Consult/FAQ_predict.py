#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 09:18
# @Author  : Adolf
# @Site    : 
# @File    : FAQ_predict.py
# @Software: PyCharm
from loguru import logger
from haystack.pipelines import FAQPipeline

from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import ElasticsearchDocumentStore


class FAQPredict:
    def __init__(self):
        document_store = ElasticsearchDocumentStore(
            host="localhost",
            port=9200,
            index="xz",
            embedding_field="query_emb",
            embedding_dim=768,
            excluded_meta_data=["query_emb"],
            similarity="cosine",
        )

        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="model/similarity_model/simcse-model-tax-market",
            model_format="sentence_transformers",
            use_gpu=False,
            scale_score=False,
        )

        self.pipe = FAQPipeline(retriever=retriever)

    def __call__(self, text):
        prediction = self.pipe.run(query=text, params={"Retriever": {"top_k": 1}})["answers"]
        return_prediction = prediction[0]

        score = return_prediction.to_dict()["score"]
        meta = return_prediction.meta
        query = meta["query"]
        answer = meta["answer"]
        # source = meta["source"]
        logger.info(f"text:{text}")
        logger.debug(f"query:{query},answer:{answer},score:{score}")
        # logger.debug(f"query:{query}, answer:{answer}, source:{source}, score:{score}")

        return answer


if __name__ == '__main__':
    m = FAQPredict()
    m("何谓耐受性？")
