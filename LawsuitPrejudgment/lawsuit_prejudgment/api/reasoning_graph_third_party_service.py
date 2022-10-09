# -*- coding: utf-8 -*-
import json
import traceback
import logging
import logging.handlers

import pandas as pd
import requests
from flask import Flask
from flask import request

from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.applicable_law_dto import \
    CriminalApplicableLawDictCreator
from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.prejudgment_report_dto import CivilReportDTO, \
    AdministrativeReportDTO, CriminalReportDTO
from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.similar_case_dto import \
    CriminalSimilarCaseListCreator
from LawsuitPrejudgment.lawsuit_prejudgment.constants import SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH, \
    CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, CIVIL_PROBLEM_TEMPLATE_CONFIG_PATH, FEATURE_TOGGLES_CONFIG_PATH
from LawsuitPrejudgment.lawsuit_prejudgment.core import civil_similar_case
from LawsuitPrejudgment.lawsuit_prejudgment.core.civil_juding_rule import CivilJudgingRule
from LawsuitPrejudgment.lawsuit_prejudgment.core.civil_relevant_law import CivilRelevantLaw
from LawsuitPrejudgment.lawsuit_prejudgment.core.civil_similar_case import CivilSimilarCase, \
    ManuallySelectedCivilSimilarCase
from LawsuitPrejudgment.lawsuit_prejudgment.feature_toggles import FeatureToggles
from Utils.io import read_json_attribute_value
from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn
from LawsuitPrejudgment.Administrative.administrative_api_v1 import *
from Utils.http_response import response_successful_result, response_failed_result
from LawsuitPrejudgment.common.config_loader import user_ps


"""
推理图谱的接口
"""
app = Flask(__name__)
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
logger.setLevel(logging.INFO)
handler = logging.handlers.TimedRotatingFileHandler('./service.log', when='D', interval=1)
handler.setFormatter(formatter)
logger.addHandler(handler)


def _request_parse(_request):
    '''解析请求数据并以json形式返回'''
    if _request.method == 'POST':
        return _request.json
    elif _request.method == 'GET':
        return _request.args
    else:
        raise Exception("传入了不支持的方法。")


@app.route('/get_civil_problem_summary', methods=["get"])
def get_civil_problem_summary():
    try:
        with open("LawsuitPrejudgment/main/civil_problem_summary.json") as json_data:
            problem_summary = json.load(json_data)["value"]
        return json.dumps({"success": True, "error_msg": "", "value": problem_summary}, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"success": False, "error_msg": repr(e), "value": None}, ensure_ascii=False)

def _get_problem(problem_id):
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("problem") for item in problem_id_mapping_list if str(item.get("id")) == str(problem_id)), None)

def _get_mapped_problem_id(problem_id):
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("mapped_id") for item in problem_id_mapping_list if str(item.get("id")) == str(problem_id)), None)

def _get_mapped_problem_id_by_name(problem):
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("mapped_id") for item in problem_id_mapping_list if str(item.get("problem")) == str(problem)), None)


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


@app.route('/get_template_by_problem_id', methods=["get"])
def get_template_by_problem_id():
    problem = _get_problem(request.args.get("problem_id"))
    df = pd.read_csv("LawsuitPrejudgment/main/用户描述案例.csv", encoding="utf-8")

    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": {
            "template": next((row["描述模板"] for index, row in df.iterrows() if str(row["案由名称"]).strip() == str(problem).strip()), "")
        }}, ensure_ascii=False)


@app.route('/get_claim_list_by_problem_id', methods=["get", "post"])
def get_claim_list_by_problem_id():
    req_data = _request_parse(request)

    # 从诉求配置，获取诉求列表
    mapped_problem = _get_mapped_problem(req_data.get("problem_id"))
    claim_list = user_ps.get(str(mapped_problem), [])

    # 从纠纷展示配置，获取诉求列表
    displayed_problem_name = _get_problem(req_data.get("problem_id"))
    df = pd.read_csv("LawsuitPrejudgment/lawsuit_prejudgment/src/纠纷展示配置.csv", encoding="utf-8")
    df = df["对应诉求"].groupby(df["纠纷展示名称"], sort=False).agg(lambda x: list(x))
    displayed_claim_list = df[displayed_problem_name] if displayed_problem_name in df else claim_list

    # 移除诉求配置中没有的诉求，以免问答产生异常
    displayed_claim_list = [item for item in displayed_claim_list if item in claim_list]

    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": [{"id": idx, "claim": claim, "is_recommended": False} for idx, claim in enumerate(displayed_claim_list)]
    }, ensure_ascii=False)


@app.route('/get_claim_by_claim_id', methods=["get"])
def get_claim_by_claim_id():
    claim_id = str(request.args.get("claim_id"))
    # mock data
    mapping = {
        "461": "请求离婚",
        "462": "请求分割财产",
        "463": "请求返还彩礼"
    }
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": {
            "claim": mapping.get(claim_id)
        }
    }, ensure_ascii=False)
    pass


# def _mapping_problem(problem):
#     mapping = {
#         "财产分割": "婚姻继承",
#         "同居问题": "婚姻继承",
#         "婚姻家庭": "婚姻继承",
#         "继承纠纷": "婚姻继承",
#         "子女抚养": "婚姻继承",
#         "老人赡养": "婚姻继承",
#         "返还彩礼": "婚姻继承"
#     }
#     return mapping.get(problem, problem)


@app.route('/reasoning_graph_result', methods=["post"])  # "service_type":'ft'
def reasoning_graph_result():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            problem = in_dict['problem']
            problem_name_for_search = problem
            mapped_problem_id = _get_mapped_problem_id_by_name(problem)
            problem = _get_mapped_problem(problem, "problem")
            problem_name_for_search = str(problem_name_for_search) + " " + str(problem)
            claim_list = in_dict['claim_list']
            fact = in_dict.get('fact', '')
            question_answers = in_dict.get('question_answers', {})
            factor_sentence_list = in_dict.get('factor_sentence_list', [])

            logging.info("=============================================================================")
            logging.info("1.problem: %s" % (problem))
            logging.info("2.claim_list: %s" % (claim_list))
            logging.info("3.fact: %s" % (fact))
            logging.info("4.question_answers: %s" % (question_answers))

            result_dict = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)

            logging.info("5.result.result_dict: %s" % (result_dict))

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
                    manually_selected_civil_similar_case = ManuallySelectedCivilSimilarCase(problem_claim.split("_")[0], problem_claim.split("_")[1], logic_tree.get_situation())
                    similar_case.extend(manually_selected_civil_similar_case.get_similar_cases())
                civilSimilarCase = CivilSimilarCase(fact, problem_name_for_search, claim_list, mapped_problem_id)
                similar_case.extend(civilSimilarCase.get_similar_cases())
                similar_case = civil_similar_case.sort_similar_cases(similar_case)

                # 获取裁判规则
                civil_judging_rule = CivilJudgingRule(fact, problem)
                judging_rule = civil_judging_rule.get_judging_rules()

                result = parsed_result
            logging.info("6.service.result: %s" % (result))
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
            return json.dumps(response_dict, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error_msg": "request data is none."
            }, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


def _get_supported_administrative_types():
    return read_json_attribute_value(SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH, "supported_administrative_types")


@app.route('/get_administrative_type', methods=["get"])
def get_administrative_type():
    supported_administrative_types = _get_supported_administrative_types()
    return response_successful_result(supported_administrative_types)


@app.route('/get_administrative_problem_and_situation_by_type_id', methods=["get", "post"])
def get_administrative_problem_and_situation_by_type_id():
    try:
        req_data = _request_parse(request)
        administrative_type = req_data.get("type_id")
        situation_dict = get_administrative_prejudgment_situation(administrative_type)
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

        return json.dumps({
            "success": True,
            "error_msg": "",
            "result": result,
        }, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


@app.route('/get_administrative_result', methods=["post"])
def get_administrative_result():
    try:
        req_data = _request_parse(request)
        administrative_type = req_data.get("type_id")
        situation = req_data.get("situation")
        res = get_administrative_prejudgment_result(administrative_type, situation)
        if FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH).reformat_prejudgment_report:
            res = AdministrativeReportDTO(res).to_dict()
        return response_successful_result(res)
    except Exception as e:
        logging.info(traceback.format_exc())
        return response_failed_result("unknown error:" + repr(e))


@app.route('/get_criminal_result', methods=["post"])
def get_criminal_result():
    try:
        url = "http://127.0.0.1:5080/get_criminal_result"
        body = {
            "fact": request.json.get("fact", ""),
            "question_answers": request.json.get("question_answers", {}),
            "factor_sentence_list": request.json.get("factor_sentence_list", []),
            "anyou": request.json.get("anyou"),
            "event": request.json.get("event")
        }
        resp_json = requests.post(url, json=body).json()
        resp_json = CriminalReportDTO(resp_json).to_dict()
        print("###########Result########")
        print(resp_json)
        return json.dumps(resp_json, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, debug=False)  # , use_reloader=False)