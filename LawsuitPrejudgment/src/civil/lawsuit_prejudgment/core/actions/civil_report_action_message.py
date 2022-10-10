#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 16:05 
@Desc    : None
"""
from dataclasses import dataclass
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.core.actions.action_message import ActionMessage


@dataclass(frozen=True)
class CivilReportActionMessage(ActionMessage):
    """ 产生民事报告需要的数据，包括案由、诉求、识别的情形、用户描述等。"""
    problem: str
    claim: str
    situation: str
    fact: str
