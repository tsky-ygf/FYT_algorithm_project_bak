#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 14:19 
@Desc    :
    功能开关的实现类。
    代码参考:https://github.com/vwt-digital/feature-toggles/blob/develop/demo/main.py
"""
from LawsuitPrejudgment.lawsuit_prejudgment.shared.utils.base_feature_toggle import BaseFeatureToggle


class FeatureToggles(BaseFeatureToggle):
    http_situation_classifier: bool
