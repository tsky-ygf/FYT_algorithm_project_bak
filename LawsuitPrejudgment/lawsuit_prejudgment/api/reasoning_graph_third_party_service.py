# -*- coding: utf-8 -*-
import json
import traceback
import logging
import logging.handlers
import requests
from flask import Flask
from flask import request

from LawsuitPrejudgment.lawsuit_prejudgment.constants import SUPPORTED_ADMINISTRATIVE_TYPES_CONFIG_PATH
from Utils.io import read_json_attribute_value
from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn
from LawsuitPrejudgment.Administrative.administrative_api_v1 import *
from Utils.http_response import response_successful_result

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


@app.route('/get_template_by_problem_id', methods=["get"])
def get_template_by_problem_id():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": {
            "template": "男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。"
        }}, ensure_ascii=False)
    pass


@app.route('/get_claim_list_by_problem_id', methods=["get"])
def get_claim_list_by_problem_id():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": [{
            "id": 461,
            "claim": "请求离婚"
        }, {
            "id": 462,
            "claim": "请求分割财产"
        }, {
            "id": 463,
            "claim": "请求返还彩礼"
        }]
    }, ensure_ascii=False)
    pass


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


def _mapping_problem(problem):
    mapping = {
        "财产分割": "婚姻继承",
        "同居问题": "婚姻继承",
        "婚姻家庭": "婚姻继承",
        "继承纠纷": "婚姻继承",
        "子女抚养": "婚姻继承",
        "老人赡养": "婚姻继承",
        "返还彩礼": "婚姻继承"
    }
    return mapping.get(problem, problem)


@app.route('/reasoning_graph_result', methods=["post"])  # "service_type":'ft'
def reasoning_graph_result():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            problem = in_dict['problem']
            problem = _mapping_problem(problem)
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
            else:
                parsed_result = []
                for suqiu, report in result.items():
                    parsed_result.append({
                        "claim": suqiu,
                        "support_or_not": report.get("support_or_not"),
                        "possibility_support": report.get("possibility_support"),
                        "reason_of_evaluation": report.get("reason_of_evaluation"),
                        "evidence_module": report.get("evidence_module"),
                        "legal_advice": report.get("legal_advice"),
                        "applicable_law": [{
                            "law_name": "《中华人民共和国民法典》",
                            "law_item": "第一千零八十九条",
                            "law_content": "离婚时,夫妻共同债务应当共同偿还。共同财产不足清偿或者财产归各自所有的，由双方协议清偿;协议不成的，由人民法院判决。"
                        },
                            {
                                "law_name": "《最高人民法院关于适用《中华人民共和国婚姻法》若干问题的解释(二)》",
                                "law_item": "第十条",
                                "law_content": "当事人请求返还按照习俗给付的彩礼的，如果查明属于以下情形，人民法院应当予以支持：（一）双方未办理结婚登记手续的；（二）双方办理结婚登记手续但确未共同生活的；（三）婚前给付并导致给付人生活困难的。适用前款第（二）、（三）项的规定，应当以双方离婚为条件。"
                            }
                        ],
                        "similar_case": [{
                            "doc_id": "2b2ed441-4a86-4f7e-a604-0251e597d85e",
                            "similar_rate": 0.88,
                            "title": "原告王某某与被告郝某某等三人婚约财产纠纷一等婚约财产纠纷一审民事判决书",
                            "court": "公主岭市人民法院",
                            "judge_date": "2016-04-11",
                            "case_number": "（2016）吉0381民初315号",
                            "tag": "彩礼 证据 结婚 给付 协议 女方 当事人 登记 离婚",
                            "win_or_not": True
                        },
                            {
                                "doc_id": "ws_c4b1e568-b253-4ac3-afd7-437941f1b17a",
                                "similar_rate": 0.80,
                                "title": "原告彭华刚诉被告王金梅、王本忠、田冬英婚约财产纠纷一案",
                                "court": "龙山县人民法院",
                                "judge_date": "2011-07-12",
                                "case_number": "（2011）龙民初字第204号",
                                "tag": "彩礼 酒席 结婚 费用 订婚 电视 女方 买家 猪肉",
                                "win_or_not": False
                            }
                        ]
                    })
                result = parsed_result
            logging.info("6.service.result: %s" % (result))
            return json.dumps({
                "success": True,
                "error_msg": "",
                "question_asked": question_asked,
                "question_next": question_next,
                "question_type": question_type,
                "factor_sentence_list": factor_sentence_list,
                "result": result
            }, ensure_ascii=False)
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
        return json.dumps({
            "success": True,
            "error_msg": "",
            "result": res,
        }, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


def _construct_response_format(resp_json):
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
        "imprisonment": int(resp_json.get("imprisonment"))
    }
    return result


def _get_criminal_report(fact):
    # 调用刑事预判的接口，获取结果
    url = "http://172.19.82.198:5060/get_criminal_report"
    data = {
        "question": fact
    }
    resp_json = requests.post(url, json=data).json()
    return _construct_response_format(resp_json)


class CriminalDemo:
    def __init__(self):
        self.demo_fact = "2020年7、8月份的一天，小黄电话联系我要买一小包毒品，我们约好当天下午3点在杭州市郊区某小区附近碰头。当天下午我们碰头后，我将一小包毒品塞给了小黄，收了他1500元，然后我们就各自回去了。"
        self.first_question = "请问贩卖的毒品是以下哪种类型？:冰毒;海洛因;鸦片;其他"
        self.next_question_dict = {
            "请问贩卖的毒品是以下哪种类型？:冰毒;海洛因;鸦片;其他": "请问贩卖的毒品数量有多少克？",
            "请问贩卖的毒品数量有多少克？": None
        }
        self.question_type_dict = {
            "请问贩卖的毒品是以下哪种类型？:冰毒;海洛因;鸦片;其他": "1",
            "请问贩卖的毒品数量有多少克？": "0"
        }

    def is_demo(self, fact):
        return fact == self.demo_fact

    def get_next_question(self, fact, question_answers):
        if not self.is_demo(fact):
            return None
        if not question_answers:
            return self.first_question

        last_asked_question = list(question_answers.keys())[-1]
        return self.next_question_dict.get(last_asked_question)

    def get_question_type(self, question):
        return self.question_type_dict.get(question, "1")

    def successful_response(self, fact, question_answers, factor_sentence_list):
        next_question = self.get_next_question(fact, question_answers)
        return json.dumps({
            "success": True,
            "error_msg": "",
            "question_asked": question_answers,
            "question_next": next_question,
            "question_type": self.get_question_type(next_question),
            "factor_sentence_list": [],
            "result": None if next_question else _get_criminal_report(fact)
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
    app.run(host="0.0.0.0", port=5090, debug=True)  # , use_reloader=False)
