#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/24 16:30
# @Author  : Adolf
# @Site    :
# @File    : basic_prejudgment.py
# @Software: PyCharm
import sys
from loguru import logger
from dataclasses import dataclass


@dataclass
class PrejudgmentConfig:
    log_level: str = "INFO"
    log_path: str = ""
    prejudgment_type: str = ""
    anyou_identify_model_path: str = ""
    situation_identify_model_path: str = ""


class PrejudgmentPipeline:
    def __init__(self, *args, **kwargs) -> None:
        self.content = dict()
        self.config = PrejudgmentConfig(*args, **kwargs)

        self.logger = logger
        self.logger.remove()  # 删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
        self.logger.add(sys.stderr, level=self.config.log_level.upper())  # 添加一个终端输出的内容
        # 添加一个文件输出的内容
        if self.config.log_path != "":
            self.logger.remove()
            logger.add(self.config.log_path, rotation='0:00', enqueue=True, retention="10 days")

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

    def __call__(self, **kwargs):
        self.content.update(kwargs)
        if "question_answers" not in self.content:
            self.content["question_answers"] = dict()

        if "anyou" not in self.content:
            print("父类：预测案由！")
            self.anyou_identify()

        if "suqiu" not in self.content:
            self.suqiu_identify()

        if "base_logic_graph" not in self.content:
            self.parse_config_file()

        if "event" not in self.content:
            print("父类：模型抽取！")
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
