#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 15:58 
@Desc    : None
"""
from xmindparser import xmind_to_dict

from LawsuitPrejudgment.lawsuit_prejudgment.constants import KNOWLEDGE_FILE_PATH
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.action import Action
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.civil_report_action_message import CivilReportActionMessage


def _get_xmind_content(problem, claim, situation):
    xmind_dict = xmind_to_dict(KNOWLEDGE_FILE_PATH + problem + '/' + problem + '_' + claim + '.xmind')[0]['topic']
    situation_node = next(node for node in xmind_dict["topics"][0]["topics"] if node["title"] == situation)
    simple_result_node = situation_node["topics"][0]
    conclusion_node = simple_result_node["topics"][0]
    advice_node = conclusion_node["topics"][0]

    return {
        "situation": situation_node["title"],
        "support_or_not": simple_result_node["title"],
        "reason_of_evaluation": conclusion_node["title"],
        "legal_advice": advice_node["title"]
    }


def _get_civil_report(problem, claim, situation, fact):
    report_dict = dict()
    result_dict = dict()

    # get data
    xmind_content = _get_xmind_content(problem, claim, situation)
    proof = ""  # TODO: 怎么处理证据
    possibility_support = 0.88

    # construct result
    prob_suqiu = claim
    report_dict[prob_suqiu] = {
        "reason_of_evaluation": xmind_content["reason_of_evaluation"],
        "evidence_module": proof,
        "legal_advice": xmind_content["legal_advice"],
        "possibility_support": possibility_support,
        "support_or_not": xmind_content["support_or_not"]
    }

    result_dict['question_asked'] = dict()  # 问过的问题
    result_dict['question_next'] = None  # 下一个要问的问题
    result_dict['question_type'] = "1"  # 下一个要问的问题的类型
    result_dict['factor_sentence_list'] = []  # 匹配到短语的列表，去重
    result_dict['result'] = report_dict  # 评估报告，包括评估理由、证据模块、法律建议、支持与否

    return result_dict


class CivilReportAction(Action):
    """ 产生民事预判的报告 """

    def run(self, message: CivilReportActionMessage):
        return _get_civil_report(message.problem, message.claim, message.situation, message.fact)
