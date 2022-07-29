#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 15:09 
@Desc    : HttpBasedSituationClassifier的测试程序。
"""
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.http_based_situation_classifier import HttpClient, \
    HttpBasedSituationClassifier
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier_message import \
    SituationClassifierMessage


def test_classify_situations():
    http_client = HttpClient(url="http://172.19.82.199:7998/situationreview")
    message = SituationClassifierMessage(suqiu="财产分割",
                                         fact="原告袁某向本院提出诉讼请求：1、依法分割原被告婚姻存续期间共同所有三个公司的财产，财产数额130万元，以结算价评估为准；"
                                              "2、判令被告承担本案诉讼费用。事实和理由：原被告于××××年××月登记结婚，婚姻存续期间共同成立了连云港顺祥环保科技有限公司"
                                              "（以下简称顺祥公司）、连云港惠某机械制造有限公司（以下简称连云港惠某公司）、如皋市惠某机械制造有限公司（以下简称如皋惠某公司）"
                                              "。因感情破裂，于2019年4月9日登记离婚，协议离婚时仅对子女抚养问题进行安排和房屋、存款等部分财产进行处理，"
                                              "双方并未对上述三个公司进行分割。现双方无法达成一致处理意见，特向贵院起诉，请求支持原告诉求。")

    situation_classifier = HttpBasedSituationClassifier(http_client)
    resp_json = situation_classifier.classify_situations(message)

    print(resp_json)
    assert resp_json
