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

        result = [
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
            for item in self.response_dict["result"]
        ]
        self.response_dict["result"] = result
        return self.response_dict
