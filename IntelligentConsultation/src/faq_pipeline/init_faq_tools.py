#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 11:57
# @Author  : Adolf
# @Site    : 
# @File    : init_faq_tools.py
# @Software: PyCharm
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever

__all__ = ["init_document_store", "init_retriever"]


def init_document_store(index_name):
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        port=9200,
        index=index_name,
        embedding_field="query_emb",
        embedding_dim=768,
        excluded_meta_data=["query_emb"],
        similarity="cosine",
    )
    return document_store


def init_retriever(document_store, model_name):
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=model_name,
        model_format="sentence_transformers",
        use_gpu=False,
        scale_score=False,
    )
    return retriever
