#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 19/9/2022 17:26 
@Desc    : None
"""
import re

import pandas as pd

df_judging_rule = pd.read_csv("LawsuitPrejudgment/lawsuit_prejudgment/core/裁判规则.csv", encoding="utf-8")
black_list = ["行政", "刑事", "公司法", "民事诉讼程序", "执行"]


class CivilJudgingRule:
    def __init__(self, fact, problem):
        self.fact = fact
        self.problem = problem

    def get_judging_rules(self):
        result = []
        for index, row in df_judging_rule.iterrows():
            if (row["纠纷类型"] in black_list) or (self.problem not in row["纠纷类型"]):
                continue
            if re.findall(row["关键词"], self.fact):
                result.append(
                    {
                        "rule_id": row["uq_id"],
                        "judging_rule": row["裁判规则"],
                        "source": row["source"],
                        "source_url": row["url"]
                    }
                )
        return result


if __name__ == '__main__':
    civil_judging_rule = CivilJudgingRule("对方经常酗酒，我想和他离婚。", "婚姻继承")
    print(civil_judging_rule.get_judging_rules())