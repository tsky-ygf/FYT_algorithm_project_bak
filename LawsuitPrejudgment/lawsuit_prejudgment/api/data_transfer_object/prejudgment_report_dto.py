#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/8/2022 17:17 
@Desc    : None
"""


class CivilReportDTO:
    def __init__(self, response_dict):
        self.response_dict = response_dict

    def to_dict(self):
        if self.response_dict.get("question_next"):
            return self.response_dict

        report = [
            [
                {
                    "type": "TYPE_TEXT",
                    "title": "诉求",
                    "content": item["claim"]
                },
                {
                    "type": "TYPE_TEXT",
                    "title": "结论",
                    "content": item["support_or_not"]
                },
                {
                    "type": "TYPE_GRAPH_OF_PROB",
                    "title": "支持概率",
                    "content": item["possibility_support"]
                },
                {
                    "type": "TYPE_TEXT",
                    "title": "评估理由",
                    "content": item["reason_of_evaluation"]
                },
                {
                    "type": "TYPE_TEXT",
                    "title": "证据材料",
                    "content": item["evidence_module"]
                },
                {
                    "type": "TYPE_TEXT",
                    "title": "法律建议",
                    "content": item["legal_advice"]
                }
            ]
            for item in self.response_dict["result"]["report"]
        ]
        self.response_dict["result"]["report"] = report
        return self.response_dict


class AdministrativeReportDTO:
    def __init__(self, response_dict):
        self.response_dict = response_dict

    def to_dict(self):
        applicable_law = self.response_dict["applicable_law"]
        similar_case = self.response_dict["similar_case"]
        judging_rule = self.response_dict["judging_rule"]
        report = [
            [
                {
                    "type": "TYPE_TEXT",
                    "title": "具体情形",
                    "content": self.response_dict["specific_situation"]["content"]
                },
                {
                    "type": "TYPE_LIST_OF_TEXT",
                    "title": "涉嫌违法行为",
                    "content": self.response_dict["suspected_illegal_act"]["content"]
                },
                {
                    "type": "TYPE_LIST_OF_OBJECT",
                    "title": "法条依据",
                    "content": self.response_dict["legal_basis"]["content"]
                },
                {
                    "type": "TYPE_LIST_OF_TEXT",
                    "title": "处罚种类",
                    "content": self.response_dict["punishment_type"]["content"]
                },
                {
                    "type": "TYPE_LIST_OF_TEXT",
                    "title": "punishment_range",
                    "content": self.response_dict["punishment_type"]["content"]
                },
                {
                    "type": "TYPE_LIST_OF_OBJECT",
                    "title": "涉刑风险",
                    "content": self.response_dict["criminal_risk"]["content"]
                }
            ]
        ]
        self.response_dict = {
            "applicable_law": applicable_law,
            "similar_case": similar_case,
            "judging_rule": judging_rule,
            "report": report
        }
        return self.response_dict
