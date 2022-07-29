#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 15:47 
@Desc    : None
"""
import abc

from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.action_message import ActionMessage


class Action(abc.ABC):
    """ 抽象基类：用于执行动作。"""

    @abc.abstractmethod
    def run(self, message: ActionMessage):
        pass
