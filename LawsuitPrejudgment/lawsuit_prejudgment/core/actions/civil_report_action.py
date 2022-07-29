#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 15:58 
@Desc    : None
"""
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.action import Action
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.civil_report_action_message import CivilReportActionMessage


def _get_civil_report(problem, claim, situation, fact):
    report_dict = dict()
    result_dict = dict()

    # mock data
    result = 1
    reason = "测试理由"
    proof = "测试证据"
    advice = "测试建议"
    possibility_support = 0.88

    # result
    support_or_not = '支持' if result == 1 else '不支持'
    prob_suqiu = claim
    report_dict[prob_suqiu] = {'reason_of_evaluation': reason,
                               'evidence_module': proof,
                               'legal_advice': advice,
                               'possibility_support': possibility_support,
                               'support_or_not': support_or_not}

    result_dict['question_asked'] = dict()  # 问过的问题
    result_dict['question_next'] = None  # 下一个要问的问题
    result_dict['question_type'] = "1"  # 下一个要问的问题的类型
    result_dict['factor_sentence_list'] = []  # 匹配到短语的列表，去重
    result_dict['result'] = report_dict  # 评估报告，包括评估理由、证据模块、法律建议、支持与否

    return result_dict
    pass


class CivilReportAction(Action):
    """ 产生民事预判的报告 """
    def run(self, message: CivilReportActionMessage):
        return _get_civil_report(message.problem, message.claim, message.situation, message.fact)
