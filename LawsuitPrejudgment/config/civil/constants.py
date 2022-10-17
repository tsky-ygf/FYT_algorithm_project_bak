#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/1 16:37 
@Desc    : None
"""
# 功能开关配置路径
FEATURE_TOGGLES_CONFIG_PATH = "LawsuitPrejudgment/config/civil/toggles.yaml"
# HttpSituationClassifier支持的案由
HTTP_SITUATION_CLASSIFIER_SUPPORT_PROBLEMS_CONFIG_PATH = "LawsuitPrejudgment/src/civil/lawsuit_prejudgment/nlu/situation_classifiers/http_classifier_support_problems.json"
# 行政预判支持的案由
SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH = "LawsuitPrejudgment/config/administrative/supported_administrative_types.json"
# 知识文件的路径
KNOWLEDGE_FILE_PATH = "data/LawsuitPrejudgment/config/"
# 模型路径
MODEL_FILE_PATH = "model/LawsuitPrejudgment/"
# 刑事相似案例服务的URL
CRIMINAL_SIMILIARITY_URL = "http://172.19.82.198:5061/criminal_similiarity"
# 民事预判前端展示的纠纷类型
CIVIL_PROBLEM_SUMMARY_CONFIG_PATH = "LawsuitPrejudgment/config/civil/civil_problem_summary.json"
# 民事案由id映射关系(页面展示的案由及id->实际的案由及id)
CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH = "LawsuitPrejudgment/config/civil/civil_problem_id_mapping.json"
# 民事纠纷经过描述模板
CIVIL_PROBLEM_TEMPLATE_CONFIG_PATH = "LawsuitPrejudgment/config/civil/用户描述案例.csv"
# 民事预判相似案例的id前缀
CIVIL_SIMILAR_CASE_ID_PREFIX = "_CIVIL_"