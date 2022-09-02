# -*- coding: utf-8 -*-
import json
import traceback
import logging
import logging.handlers
import requests
from flask import Flask
from flask import request

from LawsuitPrejudgment.Criminal.criminal_prejudgment import CriminalPrejudgment
from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.applicable_law_dto import \
    CriminalApplicableLawDictCreator
from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.prejudgment_report_dto import CivilReportDTO, \
    AdministrativeReportDTO, CriminalReportDTO
from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.similar_case_dto import \
    CriminalSimilarCaseListCreator
from LawsuitPrejudgment.lawsuit_prejudgment.constants import SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH, \
    CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, CIVIL_PROBLEM_TEMPLATE_CONFIG_PATH, FEATURE_TOGGLES_CONFIG_PATH
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


def _get_mapped_problem_id(problem_id):
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("mapped_id") for item in problem_id_mapping_list if str(item.get("id")) == str(problem_id)),
                None)


def _get_mapped_problem(attribute_value, attribute_name="id"):
    """

    Args:
        attribute_value: problem_id or problem
        attribute_name:"id" or "problem"

    Returns:
        mapped problem.
    """
    problem_id_mapping_list = read_json_attribute_value(CIVIL_PROBLEM_ID_MAPPING_CONFIG_PATH, "value")
    return next((item.get("mapped_problem") for item in problem_id_mapping_list if
                 str(item.get(attribute_name)) == str(attribute_value)), None)


@app.route('/get_template_by_problem_id', methods=["get"])
def get_template_by_problem_id():
    mapped_problem_id = _get_mapped_problem_id(request.args.get("problem_id"))
    civil_problem_template_dict = read_json_attribute_value(CIVIL_PROBLEM_TEMPLATE_CONFIG_PATH, "value")

    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": {
            "template": civil_problem_template_dict.get(str(mapped_problem_id), "")
        }}, ensure_ascii=False)


@app.route('/get_claim_list_by_problem_id', methods=["get"])
def get_claim_list_by_problem_id():
    mapped_problem = _get_mapped_problem(request.args.get("problem_id"))
    claim_list = user_ps.get(str(mapped_problem), [])
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": [{"id": idx, "claim": claim} for idx, claim in enumerate(claim_list)]
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
            problem = _get_mapped_problem(problem, "problem")
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
                applicable_law = [{
                    "law_id": "zhong-hua-ren-min-gong-he-guo-min-fa-dian-di-yi-qian-ling-ba-shi-jiu-tiao",
                    "law_name": "《中华人民共和国民法典》",
                    "law_item": "第一千零八十九条",
                    "law_content": "离婚时,夫妻共同债务应当共同偿还。共同财产不足清偿或者财产归各自所有的，由双方协议清偿;协议不成的，由人民法院判决。"
                },
                    {
                        "law_id": "zui-gao-ren-min-fa-yuan-guan-yuy-shi-yong-zhong-hua-ren-min-gong-he-guo-hun-yin-fa-ruo-gan-wen-ti-de-jie-shi-er-di-shi-tiao",
                        "law_name": "《最高人民法院关于适用《中华人民共和国婚姻法》若干问题的解释(二)》",
                        "law_item": "第十条",
                        "law_content": "当事人请求返还按照习俗给付的彩礼的，如果查明属于以下情形，人民法院应当予以支持：（一）双方未办理结婚登记手续的；（二）双方办理结婚登记手续但确未共同生活的；（三）婚前给付并导致给付人生活困难的。适用前款第（二）、（三）项的规定，应当以双方离婚为条件。"
                    }
                ]
                similar_case = [
                    {
                        "doc_id": "2b2ed441-4a86-4f7e-a604-0251e597d85e",
                        "similar_rate": 0.88,
                        "title": "原告王某某与被告郝某某等三人婚约财产纠纷一等婚约财产纠纷一审民事判决书",
                        "court": "公主岭市人民法院",
                        "judge_date": "2016-04-11",
                        "case_number": "（2016）吉0381民初315号",
                        "tag": "彩礼 证据 结婚 给付 协议 女方 当事人 登记 离婚",
                        "is_guiding_case": True
                    },
                    {
                        "doc_id": "ws_c4b1e568-b253-4ac3-afd7-437941f1b17a",
                        "similar_rate": 0.80,
                        "title": "原告彭华刚诉被告王金梅、王本忠、田冬英婚约财产纠纷一案",
                        "court": "龙山县人民法院",
                        "judge_date": "2011-07-12",
                        "case_number": "（2011）龙民初字第204号",
                        "tag": "彩礼 酒席 结婚 费用 订婚 电视 女方 买家 猪肉",
                        "is_guiding_case": False
                    }
                ]
                judging_rule = [
                    {
                        "rule_id": "rule_176",
                        "content": "男女双方共同生活时间较短，尚未建立持续稳定夫妻关系的，人民法院可以判决酌情返还彩礼。",
                        "source": "中国司法案例研究中心",
                        "source_url": "http://www5.zzu.edu.cn/fxyzx/info/1006/3513.htm"
                    }
                ]
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


def _construct_response_format(question, resp_json):
    # 编排接口返回内容的格式
    accusation = []
    for item in eval(resp_json.get("accusation")):
        for crime, prob in item.items():
            accusation.append({
                "crime": crime,
                "probability": prob
            })
    articles = []
    for item in eval(resp_json.get("articles")):
        articles.append({
            "law_name": item[0],
            "law_item": item[1],
            "crime": item[2],
            "law_content": item[3],
            "probability": item[4]
        })
    result = {
        "accusation": accusation,
        "articles": articles,
        "imprisonment": int(resp_json.get("imprisonment")),
        "similar_case": CriminalSimilarCaseListCreator.create(question),
        "applicable_law": [CriminalApplicableLawDictCreator.create(law) for law in articles],
        "judging_rule": [
            {
                "rule_id": "rule_153",
                "content": "“非法买卖”毒害性物质，是指违反法律和国家主管部门规定，未经有关主管部门批准许可，擅自购买或者出售毒害性物质的行为，并不需要兼有买进和卖出的行为。",
                "source": "中国司法案例研究中心",
                "source_url": "http://www5.zzu.edu.cn/fxyzx/info/1006/2608.htm"
            }
        ]
    }
    if FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH).reformat_prejudgment_report:
        result = CriminalReportDTO(result).to_dict()
    return result


def _get_criminal_report(fact):
    # 调用刑事预判的接口，获取结果
    url = "http://172.19.82.198:5060/get_criminal_report"
    data = {
        "question": fact
    }
    resp_json = requests.post(url, json=data).json()
    return _construct_response_format(fact, resp_json)


class CriminalDemo:
    def __init__(self):
        criminal_config = {
            "log_level": "info",
            "prejudgment_type": "criminal",
            "anyou_identify_model_path": "model/gluon_model/accusation",
            "situation_identify_model_path": "http://172.19.82.199:7777/information_result",
        }
        self.criminal_pre_judgment = CriminalPrejudgment(**criminal_config)
        self.res = None

    @property
    def is_supported(self):
        if self.res is not None and "敬请期待" in self.res.get("report_result", {}):
            return False
        return True

    def get_question_element(self, question):
        if self.res:
            for element, info in self.res.get("question_answers", dict()):
                if str(info["question"]) + str(info["answer"]).replace("|", ";") == question:
                    return element
            raise Exception("invalid question.")
        raise Exception("self.res is None.")

    @property
    def next_question(self):
        if self.res:
            for element, info in self.res.get("question_answers", dict()):
                if not info.get("usr_answer"):
                    return str(info.get("question")) + str(info.get("answer")).replace("|", ";")
            return None
        raise Exception("self.res is None.")

    @property
    def question_type(self):
        if self.res:
            for element, info in self.res.get("question_answers", dict()):
                if not info.get("usr_answer"):
                    return "1" if info.get("multiplechoice", 0) == 0 else "2"
            return "1"
        raise Exception("self.res is None.")

    @property
    def result(self):
        if self.res:
            if "report_result" not in self.res:
                return None
            if self.is_supported is False:
                return {
                    "unsupport_reason": self.res.get("report_result", "你的行为属于尚未支持的犯罪类型，正在训练优化中，敬请期待！")
                }
            return {
                "crime": "盗窃罪",
                "case_fact": "根据您的描述，2022年8月12日，在某市某区某栋某单元某层某房间内，罗某某存在携带凶器盗窃等情形，窃得现金50000元财物，盗窃金额为50000元。",
                "reason_of_evaluation": "根据您的描述，盗窃数额已满 40000元人民币，但是不满80000元人民币。",
                "legal_advice": "建议受害人发现财物被盗后及时报警、保护现场，协助警方调查。",
                "similar_case": [
                    "李春华走私、贩卖、运输、制造毒品二审刑事裁定书",
                    "董志惠走私、贩卖、运输、制造毒品复核刑事裁定书"
                ],
                "applicable_law": "根据《浙江省高级人民法院、浙江省人民检察院关于印发确定盗窃罪数额标准的通知》的规定，盗窃公私财物价值人民币三千元以上不满八万元的，认定为刑法第二百六十四条规定的“数额较大。"
            }
        raise Exception("self.res is None.")

    def recover_status(self, fact, question_answers):
        input_dict = {"fact": fact}
        self.res = self.criminal_pre_judgment(**input_dict)
        for question, answer in question_answers.items():
            element = self.get_question_element(question)
            self.res["question_answers"][element]["usr_answer"] = answer
            self.res = self.criminal_pre_judgment(**self.res)

    def successful_response(self, fact, question_answers, factor_sentence_list):
        self.recover_status(fact, question_answers)
        return json.dumps({
            "success": True,
            "error_msg": "",
            "question_asked": question_answers,
            "question_next": self.next_question,
            "question_type": self.question_type,
            "factor_sentence_list": [],
            "support": self.is_supported,
            "result": self.result
        }, ensure_ascii=False)


@app.route('/get_criminal_result', methods=["post"])
def get_criminal_result():
    try:
        fact = request.json.get("fact")
        question_answers = request.json.get("question_answers")
        factor_sentence_list = request.json.get("factor_sentence_list")

        criminal_demo = CriminalDemo()
        return criminal_demo.successful_response(fact, question_answers, factor_sentence_list)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, debug=True)  # , use_reloader=False)
