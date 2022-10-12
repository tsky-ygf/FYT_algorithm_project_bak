#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/24 16:30
# @Author  : Adolf
# @Site    :
# @File    : basic_prejudgment.py
# @Software: PyCharm
import sys
from Utils.logger import get_logger, print_run_time
# from Utils.logger import print_run_time
from dataclasses import dataclass

from typing import List


@dataclass
class PrejudgmentConfig:
    log_level: str = "INFO"
    log_path: str = ""
    prejudgment_type: str = ""
    anyou_identify_model_path: str = ""
    situation_identify_model_path: List[str] = ""


class PrejudgmentPipeline:
    def __init__(self, *args, **kwargs) -> None:
        self.content = dict()
        self.config = PrejudgmentConfig(*args, **kwargs)

        self.logger = get_logger(self.config.log_level, self.config.log_path)
        #     logger.add(self.config.log_path, rotation='0:00', enqueue=True, retention="10 days")

        self.content["graph_process"] = dict()

    def init_content(self):
        self.content = dict()
        self.content["graph_process"] = dict()

    def get_next_question(self):
        # if "question_answers" not in self.content:
        #     self.content["question_answers"] = dict()
        self.logger.debug(self.content["question_answers"])
        for key, question in self.content["question_answers"].items():
            if question["status"] == 0:
                return {key: question}
        #     print(question)
        # if self.content["question_answers"][question] == "":
        #     return question

    def anyou_identify(self, *args, **kwargs):
        raise NotImplemented

    def suqiu_identify(self, *args, **kwargs):
        raise NotImplemented

    def situation_identify(self, *args, **kwargs):
        raise NotImplemented

    def get_question(self, *args, **kwargs):
        raise NotImplemented

    def parse_config_file(self, *args, **kwargs):
        raise NotImplemented

    def match_graph(self, *args, **kwargs):
        raise NotImplemented

    def generate_report(self, *args, **kwargs):
        raise NotImplemented

    # @print_run_time
    def __call__(self, **kwargs):
        self.content.update(kwargs)
        if "question_answers" not in self.content:
            self.content["question_answers"] = dict()

        if "anyou" not in self.content:
            self.logger.debug("父类：预测案由！")
            self.anyou_identify()

        if "suqiu" not in self.content:
            self.suqiu_identify()

        if "base_logic_graph" not in self.content:
            self.parse_config_file()

        if "event" not in self.content:
            self.logger.debug("父类：模型抽取！")
            self.situation_identify()
            if "report_result" in self.content:
                return self.content
            self.match_graph()

        self.get_question()

        for key, value in self.content["graph_process"].items():
            if value == 0:
                return self.content

        self.generate_report()
        # self.logger.info(pformat(self.content))
        return self.content
