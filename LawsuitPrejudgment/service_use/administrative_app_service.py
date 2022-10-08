#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 17:47 
@Desc    : None
"""
from LawsuitPrejudgment.Administrative.administrative_api_v1 import get_administrative_prejudgment_situation, \
    get_administrative_prejudgment_result
from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.prejudgment_report_dto import \
    AdministrativeReportDTO
from LawsuitPrejudgment.lawsuit_prejudgment.constants import SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH, \
    FEATURE_TOGGLES_CONFIG_PATH
from LawsuitPrejudgment.lawsuit_prejudgment.feature_toggles import FeatureToggles
from LawsuitPrejudgment.service_use.utils import successful_response
from Utils.io import read_json_attribute_value


def _get_supported_administrative_types():
    return read_json_attribute_value(SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH, "supported_administrative_types")


def get_administrative_type():
    supported_administrative_types = _get_supported_administrative_types()
    return successful_response(supported_administrative_types)


def get_administrative_problem_and_situation_by_type_id(type_id: str):
    situation_dict = get_administrative_prejudgment_situation(type_id)
    # 编排返回参数的格式
    result = []
    for problem, value in situation_dict.items():
        situations = []
        for specific_problem, its_situations in value.items():
            situations.extend(its_situations)
        result.append({
            "problem": problem,
            "situations": situations
        })

    return successful_response(result)


def get_administrative_result(type_id, situation):
    result = get_administrative_prejudgment_result(type_id, situation)
    if FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH).reformat_prejudgment_report:
        result = AdministrativeReportDTO(result).to_dict()
    return successful_response(result)
