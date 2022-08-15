#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/15 13:23 
@Desc    : None
"""
from LawsuitPrejudgment.lawsuit_prejudgment.feature_toggles import FeatureToggles


config = \
    """
http_situation_classifier:
    enabled: false
    name: New Thing Toggler
    description: This toggles the new thing on
    """


def test_feature_toggles():
    """
        传入文件路径，
        toggles = FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH)
        或者，传入文本内容
        toggles = FeatureToggles(config)
    """

    toggles = FeatureToggles(config)
    assert not toggles.http_situation_classifier.enabled
    assert toggles.http_situation_classifier.enabled is False
