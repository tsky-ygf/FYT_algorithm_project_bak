#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/8/2022 17:17 
@Desc    : None
"""


def _reorder_claims(report):
    """调整诉求报告的顺序：如果诉求A有前置诉求P，将P的报告顺序放在A后。即先展示诉求、再展示前置诉求。"""
    # 直接反序，是最简便的方式。后续发现不妥，应调整相应的实现。
    return list(reversed(report))


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
                    "type": "TYPE_GRAPH_OF_PROB",
                    "title": "支持概率",
                    "content": round(float(item.get("possibility_support", 0.5)), 2)
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
                } if item["evidence_module"] else None,
                {
                    "type": "TYPE_TEXT",
                    "title": "法律建议",
                    "content": item["legal_advice"]
                }
            ]
            for item in self.response_dict["result"]["report"]
        ]

        # 调整诉求报告的顺序
        report = _reorder_claims(report)

        # 添加裁判规则的报告内容
        if self.response_dict["result"]["judging_rule"]:
            report.append([{
                "type": "TYPE_LIST_OF_TEXT",
                "title": "裁判规则",
                "content": [item["judging_rule"] for item in self.response_dict["result"]["judging_rule"]]
            }])

        # 删除证据为空的内容
        for claim_report in report:
            if None in claim_report:
                claim_report.remove(None)

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
                    "title": "处罚幅度",
                    "content": self.response_dict["punishment_range"]["content"]
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


class CriminalReportDTO:
    def __init__(self, response_dict):
        self.response_dict = response_dict

    def to_dict(self):
        print("############WWW########")
        print(self.response_dict)
        if self.response_dict.get("question_next"):
            return self.response_dict

        if not self.response_dict["result"]:
            return self.response_dict

        if "unsupport_reason" in self.response_dict["result"]:
            report = [
                [
                    {
                        "type": "TYPE_TEXT",
                        "title": "敬请期待",
                        "content": self.response_dict["result"]["unsupport_reason"]["敬请期待"]
                    }
                ]
            ]
        else:
            report = [
                [
                    {
                        "type": "TYPE_TEXT",
                        "title": "涉嫌罪名",
                        "content": self.response_dict["result"]["crime"]
                    },
                    {
                        "type": "TYPE_TEXT",
                        "title": "案件事实",
                        "content": self.response_dict["result"]["case_fact"]
                    },
                    {
                        "type": "TYPE_TEXT",
                        "title": "评估理由",
                        "content": self.response_dict["result"]["reason_of_evaluation"]
                    },
                    {
                        "type": "TYPE_TEXT",
                        "title": "法律建议",
                        "content": self.response_dict["result"]["legal_advice"]
                    },
                    {
                        "type": "TYPE_LIST_OF_TEXT",
                        "title": "相关类案",
                        "content": self.response_dict["result"]["similar_case"]
                    },
                    {
                        "type": "TYPE_TEXT",
                        "title": "法律依据",
                        "content": self.response_dict["result"]["applicable_law"]
                    }
                ]
            ]
        self.response_dict["result"] = {
            "applicable_law": None,
            "similar_case": None,
            "judging_rule": None,
            "report": report
        }
        return self.response_dict
