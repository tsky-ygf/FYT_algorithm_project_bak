#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/1 17:20 
@Desc    : None
"""
from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn


def test_predict_fn_should_use_http_situation_classifier():
    problem = "婚姻继承"
    claim = "请求分割财产"
    fact = "原告袁某向本院提出诉讼请求：1、依法分割原被告婚姻存续期间共同所有三个公司的财产，财产数额130万元，以结算价评估为准；"
    "2、判令被告承担本案诉讼费用。事实和理由：原被告于××××年××月登记结婚，婚姻存续期间共同成立了连云港顺祥环保科技有限公司" \
    "（以下简称顺祥公司）、连云港惠某机械制造有限公司（以下简称连云港惠某公司）、如皋市惠某机械制造有限公司（以下简称如皋惠某公司）"
    "。因感情破裂，于2019年4月9日登记离婚，协议离婚时仅对子女抚养问题进行安排和房屋、存款等部分财产进行处理，"
    "双方并未对上述三个公司进行分割。现双方无法达成一致处理意见，特向贵院起诉，请求支持原告诉求。"

    result_dict = predict_fn(problem, [claim], fact, dict(), [])

    print(result_dict)
    assert result_dict
    assert result_dict["question_next"] is None
    assert str(result_dict["result"][claim]["reason_of_evaluation"]).startswith("根据您的描述")
    pass


def test_should_have_evidence():
    """ 程序仅在有支持路径连通时，返回证据。见logic_tree.py/get_next_question()/L570-L577"""
    problem = "买卖合同"
    claim = "返还定金"
    fact = "xxxxxxx"
    question_answers = {
        "存在以下哪种情形？:买受人履行了义务;买受人不履行义务或履行义务不符合约定;出卖人不履行义务或履行义务不符合约定": "买受人不履行义务或履行义务不符合约定"
    }
    factor_sentence_list = []

    result_dict = predict_fn(problem, [claim], fact, question_answers, factor_sentence_list)
    print(result_dict)
    assert result_dict
    assert result_dict["question_next"] is None
    assert result_dict["result"][claim]["evidence_module"] == ''
