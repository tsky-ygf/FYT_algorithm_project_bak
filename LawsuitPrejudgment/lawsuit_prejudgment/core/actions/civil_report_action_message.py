#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 16:05 
@Desc    : None
"""
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.action_message import ActionMessage


class CivilReportActionMessage(ActionMessage):
    """ 产生民事报告需要的数据，包括识别的情形等。"""
    situation: str
