#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/1 16:37 
@Desc    : None
"""
# 功能开关配置路径
FEATURE_TOGGLES_CONFIG_PATH = "LawsuitPrejudgment/lawsuit_prejudgment/toggles.yaml"
# HttpSituationClassifier支持的案由
HTTP_SITUATION_CLASSIFIER_SUPPORT_PROBLEMS_CONFIG_PATH = "LawsuitPrejudgment/lawsuit_prejudgment/nlu/situation_classifiers/http_classifier_support_problems.json"
# 行政预判支持的案由
SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH = "LawsuitPrejudgment/Administrative/config/supported_administrative_types.json"
# 知识文件的路径
KNOWLEDGE_FILE_PATH = "LawsuitPrejudgment/config/"
# 模型路径
MODEL_FILE_PATH = "LawsuitPrejudgment/model/"
# 刑事相似案例服务的URL
CRIMINAL_SIMILIARITY_URL = "http://172.19.82.198:5061/criminal_similiarity"
