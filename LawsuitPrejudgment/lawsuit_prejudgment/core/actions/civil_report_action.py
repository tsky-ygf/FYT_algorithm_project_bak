#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 15:58 
@Desc    : None
"""
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.action import Action
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.civil_report_action_message import CivilReportActionMessage


class CivilReportAction(Action):
    """ 产生民事预判的报告 """
    def run(self, message: CivilReportActionMessage):
        pass
