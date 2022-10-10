#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/15 13:23 
@Desc    : None
"""
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment import FeatureToggles


config = \
    """
http_situation_classifier:
    enabled: false
    name: Http Situation Classifier
    description: 调用http服务进行情形识别。

should_not_repeat_question_item:
    enabled: true
    name: Don't Repeat Question Item in the QA Process of Civil Prejudgment.
    description:
        民事诉讼预判的问答过程中，已经回答过的问题，不应在后续的问题中出现该选项。
        注意，当开关打开时，接口也应加上相应的字段repeated_question_management，否则会状态不一致而出错。
    """


def test_feature_toggles():
    """
        传入文件路径，
        toggles = FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH)
        或者，传入文本内容
        toggles = FeatureToggles(config)
    """

    toggles = FeatureToggles(config)
    print(toggles.http_situation_classifier)
    assert not toggles.http_situation_classifier.enabled
    assert toggles.http_situation_classifier.enabled is False
