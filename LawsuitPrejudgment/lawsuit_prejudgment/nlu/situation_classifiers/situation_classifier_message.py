#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 14:53 
@Desc    : None
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SituationClassifierMessage:
    """法律情形识别任务需要的数据，包括用户诉求、纠纷经过等。"""
    suqiu: str
    fact: str

    def to_dict(self):
        return {
            "suqiu": self.suqiu,
            "content": self.fact
        }
