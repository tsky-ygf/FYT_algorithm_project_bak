# -*- coding: utf-8 -*-
import json
import traceback
import logging
import logging.handlers
from flask import Flask
from flask import request
from LawsuitPrejudgment.Administrative.administrative_api_v1 import get_administrative_prejudgment_situation
from LawsuitPrejudgment.Criminal.criminal_prejudgment import CriminalPrejudgment

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


def _get_administrative_prejudgment_result(administrative_type, situation):
    """
    获取行政处罚的预测结果
    :return:
    """
    # 获取行政处罚的预测结果
    with open('LawsuitPrejudgment/Administrative/result_show/{}_config.json'.format(administrative_type), 'r') as f1:
        info_data = json.load(f1)

    # with open('LawsuitPrejudgment/Administrative/result_show/{}_type.json'.format(administrative_type), 'r') as f2:
    #     type_data = json.load(f2)

    prejudgment_result = dict()
    prejudgment_result["具体情形"] = '{}({})'.format(situation, info_data[situation]['法条类别'])

    prejudgment_result["涉嫌违法行为"] = info_data[situation]['处罚依据']
    prejudgment_result["法条依据"] = info_data[situation]['法条依据']
    prejudgment_result["处罚种类"] = info_data[situation]['处罚种类']
    prejudgment_result["处罚幅度"] = info_data[situation]['处罚幅度']
    prejudgment_result["涉刑风险"] = info_data[situation]['涉刑风险']
    prejudgment_result["相似类案"] = info_data[situation]['相关案例']

    return prejudgment_result


@app.route('/get_administrative_type', methods=["get"])
def get_administrative_type():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "result": [{
            "type_id": "tax",
            "type_name": "税务处罚预判"
        }, {
            "type_id": "police",
            "type_name": "公安处罚预判"
        }, {
            "type_id": "transportation",
            "type_name": "道路运输处罚预判"
        }]
    }, ensure_ascii=False)
    pass


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
        res = _get_administrative_prejudgment_result(administrative_type, situation)
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
            for element, info in self.res.get("question_answers", dict()).items():
                if str(info.get("question")) + ":" + str(info.get("answer")).replace("|", ";") == question:
                    return element
            raise Exception("invalid question.")
        raise Exception("self.res is None.")

    @property
    def next_question(self):
        if self.res:
            for element, info in self.res.get("question_answers", dict()).items():
                if not info.get("usr_answer"):
                    return str(info.get("question")) + ":" + str(info.get("answer")).replace("|", ";")
            return None
        raise Exception("self.res is None.")

    @property
    def question_type(self):
        if self.res:
            for element, info in self.res.get("question_answers", dict()).items():
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
            report_result = self.res.get("report_result", dict())
            return {
                "crime": report_result.get("涉嫌罪名", ""),
                "case_fact": report_result.get("案件事实", ""),
                "reason_of_evaluation": report_result.get("评估理由", ""),
                "legal_advice": report_result.get("法律建议", ""),
                "similar_case": report_result.get("相关类案", []),
                "applicable_law": report_result.get("法律依据", "")
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
    app.run(host="0.0.0.0", port=5080, debug=True)  # , use_reloader=False)
