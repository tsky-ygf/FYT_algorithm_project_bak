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
    prejudgment_type: str = ""
    xmind_path: str = ""
    anyou_identify_model_path: str = ""
    situation_identify_model_path: str = ""


class PrejudgmentPipeline:
    def __init__(self, *args, **kwargs) -> None:
        self.content = dict()
        self.config = PrejudgmentConfig(*args, **kwargs)

        self.logger = logger
        self.logger.remove()  # 删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
        self.logger.add(sys.stderr, level=self.config.log_level.upper())  # 添加一个终端输出的内容

    def anyou_identify(self, *args, **kwargs):
        raise NotImplemented

    def suqiu_identify(self, *args, **kwargs):
        raise NotImplemented

    def situation_identify(self, *args, **kwargs):
        raise NotImplemented

    def get_question(self, *args, **kwargs):
        raise NotImplemented

    def parse_xmind(self, *args, **kwargs):
        raise NotImplemented

    def generate_report(self, *args, **kwargs):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        self.content.update(kwargs)
        self.parse_xmind()
        self.anyou_identify()
        self.suqiu_identify()
        self.situation_identify()
        self.get_question()
        self.generate_report()

        self.logger.debug(self.content)
        return self.content
