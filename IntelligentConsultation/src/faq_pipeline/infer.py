#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/12 13:59
# @Author  : Adolf
# @Site    : 
# @File    : query_cls_infer.py
# @Software: PyCharm
import pandas as pd
from jieba import analyse

from IntelligentConsultation.src.faq_pipeline.init_faq_tools import init_haystack_fqa_pipe
from IntelligentConsultation.src.query_classification import query_cls_infer
from Utils.logger import get_logger, print_run_time


class FAQPredict:
    def __init__(self,
                 level="INFO",
                 console=True,
                 logger_file=None,
                 index_name="topic_qa_test",
                 model_name="model/similarity_model/simcse-model-topic-qa",
                 cls_model="model/similarity_model/query_cls/final"):
        self.logger = get_logger(level=level, console=console, logger_file=logger_file)
        self.pipe = init_haystack_fqa_pipe(index_name=index_name, model_name=model_name)

        self.tfidf = analyse.extract_tags

        self.proper_noun_config = self.get_base_config()

        self.label_map = query_cls_infer.init_config()
        self.query_cls_tokenizers, self.query_cls_model = query_cls_infer.init_torch_model(cls_model)

        self.logger.success("加载服务完成......")

    def get_base_config(self):
        proper_noun_df = pd.read_csv("IntelligentConsultation/config/专有名词.csv")
        # self.logger.debug(proper_noun_df)
        proper_noun_config = dict(zip(proper_noun_df['专有名词'], proper_noun_df['type']))
        self.logger.debug(proper_noun_config)
        return proper_noun_config

    def pre_process_text(self, text):
        self.logger.info(f"text:{text}")

        pred_cls = query_cls_infer.get_torch_model_result(self.query_cls_tokenizers, self.query_cls_model, text)
        pred_cls = self.label_map[pred_cls[0].item()]

        if text in self.proper_noun_config:
            text += "是什么"

        keywords = self.tfidf(text, allowPOS=["n", "v", "a", "d"])
        keywords_text = "".join(keywords)
        text += keywords_text
        return text, pred_cls

    def post_process_result(self, query, predictions, source=None, sub_source=None):
        similarity_question = [{"question": p.meta["query"], "answer": p.meta["answer"]} for p in predictions]
        similarity_question_info = [{"question": p.meta["query"], "score": p.to_dict()["score"]} for p in predictions]
        self.logger.info(f"similarity_question:{similarity_question_info}")

        final_answer = "您的问题法域通暂时还理解不了，请您换个说法。"
        for prediction in predictions:
            score = prediction.to_dict()["score"]
            meta = prediction.meta
            self.logger.debug(meta)
            match_query = meta["query"]
            match_answer = meta["answer"]
            match_source = meta["source"]
            match_sub_source = meta["sub_source"]
            self.logger.success(
                f"query:{match_query},score:{score},source:{match_source},sub_source:{match_sub_source}")
            # if query_type and match_source != query_type:
            #     continue
            if source is not None and match_source != source:
                continue

            if sub_source is not None and match_sub_source != sub_source:
                continue

            proper_noun_missing = False

            for proper_noun, noun_type in self.proper_noun_config.items():
                if proper_noun in query and proper_noun not in match_query:
                    proper_noun_missing = True

            if proper_noun_missing:
                continue

            similarity_question.insert(0, {"question": match_query, "answer": match_answer})
            final_answer = match_answer
            break

        return final_answer, similarity_question

    def __call__(self, query, source=None, sub_source=None):
        query, pred_sub = self.pre_process_text(query)

        self.logger.info(f"query:{query}===>pred_sub:{pred_sub}")

        predictions = self.pipe.run(query=query, params={"Retriever": {"top_k": 20}})["answers"]

        if sub_source is None:
            sub_source = pred_sub

        answer, similarity_question = self.post_process_result(query, predictions, source, sub_source)

        return answer, similarity_question


if __name__ == '__main__':
    m = FAQPredict(level="DEBUG",
                   model_name="model/similarity_model/simcse-model-all",
                   index_name="topic_qa_test")
    _answer, _similarity_question = m("公司交不起税怎么办")

    print(_answer)
