#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 14:35
# @Author  : Adolf
# @Site    : 
# @File    : administrative_api_v1.py
# @Software: PyCharm
import json
# import pandas as pd
import typing
from pprint import pprint

from LawsuitPrejudgment.config.civil.constants import FEATURE_TOGGLES_CONFIG_PATH, \
    SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH
from LawsuitPrejudgment.src.civil.utils.feature_toggle.feature_toggles import FeatureToggles
from LawsuitPrejudgment.src.common.basic_prejudgment_v2 import PrejudgmentPipeline
from LawsuitPrejudgment.src.common.data_transfer_object.applicable_law_dto import \
    AdministrativeApplicableLawDictCreator
from LawsuitPrejudgment.src.common.data_transfer_object.prejudgment_report_dto import AdministrativeReportDTO
from LawsuitPrejudgment.src.common.data_transfer_object.similar_case_dto import \
    AdministrativeSimilarCaseDictCreator
from Utils.io import read_json_attribute_value


def get_administrative_prejudgment_situation(administrative_type):
    with open('data/administrative_config/{}_type.json'.format(administrative_type), 'r') as f2:
        type_data = json.load(f2)

    return type_data


def get_administrative_prejudgment_result(administrative_type, situation):
    """
    获取行政处罚的预测结果
    :return:
    """
    # 获取行政处罚的预测结果
    with open('data/administrative_config/{}_config.json'.format(administrative_type), 'r') as f1:
        info_data = json.load(f1)

    # with open('LawsuitPrejudgment/Administrative/result_show/{}_type.json'.format(administrative_type), 'r') as f2:
    #     type_data = json.load(f2)
    # logger.info(pformat(info_data[situation]))
    prejudgment_result = dict()

    prejudgment_result["specific_situation"] = {
        "title": "具体情形",
        "content": '{}({})'.format(situation, info_data[situation]['法条类别'])
    }
    prejudgment_result["suspected_illegal_act"] = {"title": "涉嫌违法行为", "content": info_data[situation]['处罚依据']}
    prejudgment_result["legal_basis"] = {
        "title": "法条依据",
        "content": [{"law_item": law_item, "law_content": law_content} for law_item, law_content in
                    info_data[situation]['法条依据'].items()]
    }
    prejudgment_result["punishment_type"] = {"title": "处罚种类", "content": info_data[situation]['处罚种类']}
    prejudgment_result["punishment_range"] = {"title": "处罚幅度", "content": info_data[situation]['处罚幅度']}
    prejudgment_result["criminal_risk"] = {
        "title": "涉刑风险",
        "content": [{"crime_name": crime_name, "law_item": law_info[0], "law_content": law_info[1]} for
                    crime_name, law_info in info_data[situation]['涉刑风险'].items()]
    }
    prejudgment_result["similar_case"] = [
        AdministrativeSimilarCaseDictCreator.create({"title": "相似类案", "content": content}) for content in
        info_data[situation]['相关案例']]
    law_list = prejudgment_result["legal_basis"]["content"] + prejudgment_result["criminal_risk"]["content"]
    # TODO: 去除可能重复的法条
    prejudgment_result["applicable_law"] = [AdministrativeApplicableLawDictCreator.create(law) for law in law_list]
    # TODO: mock judging rule
    prejudgment_result["judging_rule"] = [
        {
            "rule_id": "rule_189",
            "content": "案外人执行异议之诉中，查明涉案款项实体权益属案外人的，应直接判决停止对涉案款项的执行，无须以不当得利另诉。",
            "source": "越律网",
            "source_url": "https://www.sxls.com/gongbao2018.html"
        }
    ]
    return prejudgment_result


def _get_supported_administrative_types():
    return read_json_attribute_value(SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH, "supported_administrative_types")


def _get_type_id_by_anyou(anyou):
    supported_administrative_types = _get_supported_administrative_types()
    return next((item["type_id"] for item in supported_administrative_types if item["type_name"] == anyou))


def _get_anyou_list():
    supported_administrative_types = _get_supported_administrative_types()
    return [item["type_name"] for item in supported_administrative_types]


def _get_problems_by_anyou(anyou):
    type_id = _get_type_id_by_anyou(anyou)
    problem_and_situations_dict = get_administrative_prejudgment_situation(type_id)
    return [problem for problem, value in problem_and_situations_dict.items()]


def _get_situations_by_anyou_and_problem(anyou, problem):
    type_id = _get_type_id_by_anyou(anyou)
    problem_and_situations_dict = get_administrative_prejudgment_situation(type_id)
    situations = []
    value = problem_and_situations_dict[problem]
    for specific_problem, its_situations in value.items():
        situations.extend(its_situations)
    return situations


class AdministrativePrejudgment(PrejudgmentPipeline):
    def __init__(self, *args, **kwargs):
        super(AdministrativePrejudgment, self).__init__(*args, **kwargs)
        # self.dialogue_history = dict()
        # self.dialogue_state = dict()
        # self.context = {
        #     "slots": {
        #         "anyou": None,
        #         "problem": None,
        #         "situation": None
        #     }
        # }

    def recover_context(self, **kwargs):
        self.dialogue_history = kwargs["dialogue_history"]
        self.dialogue_state = kwargs["dialogue_state"]
        if kwargs["context"]:
            self.context = kwargs["context"]
        else:
            self.context = {
                "slots": {
                    "anyou": None,
                    "problem": None,
                    "situation": None
                }
            }

    def nlu(self, **kwargs):
        self.dialogue_state.claim_list = ["行政预判"]

    def _get_last_question_answer_info(self):
        if self.dialogue_history.question_answers:
            return self.dialogue_history.question_answers[-1]
        return None

    def update_context(self, **kwargs):
        last_question_answer_info = self._get_last_question_answer_info()
        if not last_question_answer_info:
            return

        filled_slot = last_question_answer_info["other"]["slot"]
        filled_value = last_question_answer_info["user_answer"]
        self.context["slots"][filled_slot] = filled_value

    def decide_next_action(self, **kwargs) -> str:
        for slot, value in self.context["slots"].items():
            if not value:
                return "ask"
        return "report"

    def _get_question_info_by_slot(self, slot):
        slot2question = {
            "anyou": "请选择你遇到的纠纷类型",
            "problem": "请选择你遇到的问题",
            "situation": "请选择具体的情形"
        }
        # TODO:
        if slot == "anyou":
            candidate_answers = _get_anyou_list()
            return {
                "question": slot2question[slot],
                "candidate_answers": candidate_answers,
                "question_type": "single",
                "other": {
                    "slot": "anyou"
                }
            }

        if slot == "problem":
            candidate_answers = _get_problems_by_anyou(self.context["slots"]["anyou"])
            return {
                "question": slot2question[slot],
                "candidate_answers": candidate_answers,
                "question_type": "single",
                "other": {
                    "slot": "problem"
                }
            }

        if slot == "situation":
            candidate_answers = _get_situations_by_anyou_and_problem(self.context["slots"]["anyou"], self.context["slots"]["problem"])
            return {
                "question": slot2question[slot],
                "candidate_answers": candidate_answers,
                "question_type": "single",
                "other": {
                    "slot": "situation"
                }
            }

    def get_next_question(self):
        for slot, value in self.context["slots"].items():
            if not value:
                return self._get_question_info_by_slot(slot)

    def generate_report(self, **kwargs):
        type_id = _get_type_id_by_anyou(self.context["slots"]["anyou"])
        situation = self.context["slots"]["situation"]
        result = get_administrative_prejudgment_result(type_id, situation)
        if FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH).reformat_prejudgment_report:
            result = AdministrativeReportDTO(result).to_dict()
        return result


if __name__ == '__main__':
    res = get_administrative_prejudgment_result("tax", "没有真实的业务、资金往来")
    pprint(res, sort_dicts=False)

    # rest = get_administrative_prejudgment_situation("tax")
    # pprint(rest, sort_dicts=False)
