#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 8/10/2022 10:02 
@Desc    : None
"""
import pandas as pd

from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.api.data_transfer_object.prejudgment_report_dto import CivilReportDTO
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.constants import CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, \
    CIVIL_PROBLEM_TEMPLATE_CONFIG_PATH, CIVIL_PROBLEM_SUMMARY_CONFIG_PATH, FEATURE_TOGGLES_CONFIG_PATH
from LawsuitPrejudgment.src.civil.common import user_ps
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.core.civil_juding_rule import CivilJudgingRule
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.core.civil_relevant_law import CivilRelevantLaw
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.core.civil_similar_case import ManuallySelectedCivilSimilarCase, \
    CivilSimilarCase, sort_similar_cases
from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.feature_toggles import FeatureToggles
from Utils.io import read_json_attribute_value
from LawsuitPrejudgment.src.civil.main.reasoning_graph_predict import predict_fn


def get_civil_problem_summary():
    problem_summary = read_json_attribute_value(CIVIL_PROBLEM_SUMMARY_CONFIG_PATH, "value")
    return {"success": True, "error_msg": "", "value": problem_summary}


def _get_problem(problem_id):
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("problem") for item in problem_id_mapping_list if str(item.get("id")) == str(problem_id)), None)


def get_template_by_problem_id(problem_id):
    problem = _get_problem(problem_id)
    df = pd.read_csv(CIVIL_PROBLEM_TEMPLATE_CONFIG_PATH, encoding="utf-8")
    template = next((row["描述模板"] for index, row in df.iterrows() if str(row["案由名称"]).strip() == str(problem).strip()), "")
    return {"success": True, "error_msg": "", "value": {"template": template}}


def _get_mapped_problem(attribute_value, attribute_name="id"):
    """

    Args:
        attribute_value: problem_id or problem
        attribute_name:"id" or "problem"

    Returns:
        mapped problem.
    """
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("mapped_problem") for item in problem_id_mapping_list if str(item.get(attribute_name)) == str(attribute_value)), None)


def get_claim_list_by_problem_id(problem_id, fact):
    # 从诉求配置，获取诉求列表
    mapped_problem = _get_mapped_problem(problem_id)
    claim_list = user_ps.get(str(mapped_problem), [])

    # 从纠纷展示配置，获取诉求列表
    displayed_problem_name = _get_problem(problem_id)
    df = pd.read_csv("LawsuitPrejudgment/config/civil/诉求展示配置.csv", encoding="utf-8")
    df = df["对应诉求"].groupby(df["纠纷展示名称"], sort=False).agg(lambda x: list(x))
    displayed_claim_list = df[displayed_problem_name] if displayed_problem_name in df else claim_list

    # 移除诉求配置中没有的诉求，以免问答产生异常
    displayed_claim_list = [item for item in displayed_claim_list if item in claim_list]

    return {
        "success": True,
        "error_msg": "",
        "value": [{"id": idx, "claim": claim, "is_recommended": False} for idx, claim in enumerate(displayed_claim_list)]
    }


def _get_mapped_problem_id_by_name(problem):
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("mapped_id") for item in problem_id_mapping_list if str(item.get("problem")) == str(problem)), None)


def reasoning_graph_result(problem, claim_list, fact, question_answers, factor_sentence_list):
    problem_name_for_search = problem
    mapped_problem_id = _get_mapped_problem_id_by_name(problem)
    problem = _get_mapped_problem(problem, "problem")
    problem_name_for_search = str(problem_name_for_search) + " " + str(problem)

    result_dict = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)

    question_asked = result_dict['question_asked']  # 问过的问题
    question_next = result_dict['question_next']  # 下一个要问的问题
    question_type = result_dict['question_type']
    factor_sentence_list = result_dict['factor_sentence_list']  # 匹配到短语的列表
    result = result_dict['result']
    if len(result) == 0:
        result = None
        applicable_law = None
        similar_case = None
        judging_rule = None
    else:
        parsed_result = []
        for suqiu, report in result.items():
            parsed_result.append({
                "claim": suqiu,
                "support_or_not": report.get("support_or_not"),
                "possibility_support": report.get("possibility_support"),
                "reason_of_evaluation": report.get("reason_of_evaluation"),
                "evidence_module": report.get("evidence_module"),
                "legal_advice": report.get("legal_advice")
            })

        # 获取相关法条
        relevant_law = CivilRelevantLaw(fact, problem, claim_list)
        applicable_law = relevant_law.get_relevant_laws()

        # 获取相似案例
        similar_case = []
        for problem_claim, logic_tree in result_dict['suqiu_tree'].items():
            manually_selected_civil_similar_case = ManuallySelectedCivilSimilarCase(problem_claim.split("_")[0],
                                                                                    problem_claim.split("_")[1],
                                                                                    logic_tree.get_situation())
            similar_case.extend(manually_selected_civil_similar_case.get_similar_cases())
        civilSimilarCase = CivilSimilarCase(fact, problem_name_for_search, claim_list, mapped_problem_id)
        similar_case.extend(civilSimilarCase.get_similar_cases())
        similar_case = sort_similar_cases(similar_case)

        # 获取裁判规则
        civil_judging_rule = CivilJudgingRule(fact, problem)
        judging_rule = civil_judging_rule.get_judging_rules()

        result = parsed_result
    response_dict = {
        "success": True,
        "error_msg": "",
        "question_asked": question_asked,
        "question_next": question_next,
        "question_type": question_type,
        "factor_sentence_list": factor_sentence_list,
        "result": {
            "applicable_law": applicable_law,
            "similar_case": similar_case,
            "judging_rule": judging_rule,
            "report": result
        } if not question_next else None
    }
    if FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH).reformat_prejudgment_report:
        response_dict = CivilReportDTO(response_dict).to_dict()
    return response_dict
