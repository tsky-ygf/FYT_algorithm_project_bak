#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 17:21 
@Desc    : None
"""
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.core.actions.civil_report_action import CivilReportAction
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.core.actions.civil_report_action_message import CivilReportActionMessage


def test_run():
    problem = "婚姻继承"
    claim = "请求分割财产"
    fact = "原告袁某向本院提出诉讼请求：1、依法分割原被告婚姻存续期间共同所有三个公司的财产，财产数额130万元，以结算价评估为准；"
    "2、判令被告承担本案诉讼费用。事实和理由：原被告于××××年××月登记结婚，婚姻存续期间共同成立了连云港顺祥环保科技有限公司" \
    "（以下简称顺祥公司）、连云港惠某机械制造有限公司（以下简称连云港惠某公司）、如皋市惠某机械制造有限公司（以下简称如皋惠某公司）"
    "。因感情破裂，于2019年4月9日登记离婚，协议离婚时仅对子女抚养问题进行安排和房屋、存款等部分财产进行处理，"
    "双方并未对上述三个公司进行分割。现双方无法达成一致处理意见，特向贵院起诉，请求支持原告诉求。"
    situation = "财产为婚姻关系存续期间夫妻的共同财产"

    message = CivilReportActionMessage(problem, claim, situation, fact)
    action = CivilReportAction()
    result = action.run(message)

    print(result)
    assert result
    assert result["result"][claim]["support_or_not"] == "支持"
    assert result["result"][claim]["reason_of_evaluation"].startswith("根据您的描述")
    assert result["result"][claim]["legal_advice"].startswith("法律建议")