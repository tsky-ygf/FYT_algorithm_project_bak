#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 13:59
# @Author  : Adolf
# @Site    : 
# @File    : infer.py
# @Software: PyCharm
import pandas as pd
from jieba import analyse

from IntelligentConsultation.src.faq_pipeline.init_faq_tools import init_haystack_fqa_pipe
from Utils.logger import get_logger


class FAQPredict:
    def __init__(self,
                 level="INFO",
                 console=True,
                 logger_file=None,
                 index_name="topic_qa_test_v2",
                 model_name="model/similarity_model/simcse-model-optim"):
        self.logger = get_logger(level=level, console=console, logger_file=logger_file)
        self.pipe = init_haystack_fqa_pipe(index_name=index_name, model_name=model_name)

        self.tfidf = analyse.extract_tags

        self.proper_noun_config = self.get_base_config()

    def get_base_config(self):
        proper_noun_df = pd.read_csv("IntelligentConsultation/config/专有名词.csv")
        # self.logger.debug(proper_noun_df)
        proper_noun_config = dict(zip(proper_noun_df['专有名词'], proper_noun_df['type']))
        self.logger.debug(proper_noun_config)
        return proper_noun_config

    def pre_process_text(self, text):
        self.logger.info(f"text:{text}")
        keywords = self.tfidf(text, allowPOS=["n", "v", "a", "d"])
        keywords_text = "".join(keywords)
        text += keywords_text * 3
        return text

    def post_process_result(self, query, predictions, query_type=None):
        similarity_question = [{"question": p.meta["query"], "answer": p.meta["answer"]} for p in predictions]
        similarity_question_info = [{"question": p.meta["query"], "score": p.to_dict()["score"]} for p in predictions]
        self.logger.info(f"similarity_question:{similarity_question_info}")

        match_answer = "您的问题法域通暂时还理解不了，请您换个说法。"
        for prediction in predictions:
            score = prediction.to_dict()["score"]
            meta = prediction.meta
            match_query = meta["query"]
            match_answer = meta["answer"]
            match_source = meta["type"]
            self.logger.success(f"query:{match_query},score:{score},source:{match_source}")
            if query_type and match_source != query_type:
                continue
            # for proper_noun in self.proper_noun_config.items():
            #     if proper_noun[0] in query and proper_noun[1] != match_source:
            # continue
            break

        return match_answer, similarity_question

    def __call__(self, query, query_type=None):
        query = self.pre_process_text(query)

        predictions = self.pipe.run(query=query, params={"Retriever": {"top_k": 10}})["answers"]

        answer, similarity_question = self.post_process_result(query, predictions, query_type)

        return answer, similarity_question


if __name__ == '__main__':
    m = FAQPredict(level="DEBUG")
    m("公司交不起税怎么办")
