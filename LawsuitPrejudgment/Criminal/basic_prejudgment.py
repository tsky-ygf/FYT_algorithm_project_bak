#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/24 16:30
# @Author  : Adolf
# @Site    : 
# @File    : basic_prejudgment.py
# @Software: PyCharm
from dataclasses import dataclass


@dataclass
class PrejudgmentConfig:
    prejudgment_type = ""
    xmind_path = ""


class PrejudgmentPipeline:
    def __init__(self):
        self.content = dict()

    def anyou_identify(self, content):
        raise NotImplemented

    def suqiu_identify(self, content):
        raise NotImplemented

    def situation_identify(self, content):
        raise NotImplemented

    def parse_xmind(self, content):
        raise NotImplemented

    def generate_report(self, content):
        raise NotImplemented

    def __call__(self):
        self.anyou_identify(self.content)
        self.suqiu_identify(self.content)
        self.situation_identify(self.content)
        self.parse_xmind(self.content)
        self.generate_report(self.content)

        return self.content
