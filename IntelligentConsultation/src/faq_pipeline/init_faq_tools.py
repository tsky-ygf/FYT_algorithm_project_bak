#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 11:57
# @Author  : Adolf
# @Site    : 
# @File    : init_faq_tools.py
# @Software: PyCharm
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever

from haystack.pipelines import FAQPipeline
from haystack import Pipeline

from haystack.nodes import TransformersQueryClassifier
from haystack.nodes.other.docs2answers import Docs2Answers


def init_document_store(index_name, recreate_index=False):
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        port=9200,
        index=index_name,
        embedding_field="query_emb",
        embedding_dim=768,
        excluded_meta_data=["query_emb"],
        similarity="cosine",
        recreate_index=recreate_index,
    )
    return document_store


def init_retriever(document_store, model_name, use_gpu=False):
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=model_name,
        model_format="sentence_transformers",
        use_gpu=use_gpu,
        scale_score=False,
    )
    return retriever


def init_haystack_fqa_pipe(index_name, model_name, use_gpu=False):
    document_store = init_document_store(index_name=index_name)
    retriever = init_retriever(document_store, model_name=model_name, use_gpu=use_gpu)
    pipe = FAQPipeline(retriever=retriever)

    # pipe = Pipeline()
    # pipe.add_node(component=query_classifier, name="QueryClassifier", inputs=["Query"])
    # pipe.add_node(component=retriever, name="Retriever", inputs=["QueryClassifier.output_1"])
    # pipe.add_node(component=Docs2Answers(), name="Docs2Answers", inputs=["Retriever"])

    return pipe


def init_query_cls():
    label_path = "IntelligentConsultation/config/query_cls_label.txt"
    label_list = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            line_list = line.strip().split(",")
            # label_list.append(line_list[0])
            label_list.append("LABEL_{}".format(line_list[1]))

    query_classifier = TransformersQueryClassifier(
        model_name_or_path="model/similarity_model/query_cls/final",
        use_gpu=False,
        labels=label_list
    )

    result = query_classifier.run(query="公司交不起税怎么办")
    print(result)
    return query_classifier


if __name__ == '__main__':
    init_query_cls()