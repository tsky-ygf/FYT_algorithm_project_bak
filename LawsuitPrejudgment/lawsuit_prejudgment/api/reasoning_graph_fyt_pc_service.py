# -*- coding: utf-8 -*-
import json
from typing import Dict
import traceback
import logging
import logging.handlers
from flask import Flask
from flask import request

from LawsuitPrejudgment.lawsuit_prejudgment.shared.utils.io import read_json_attribute_value
from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn

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


# fyt-pc-test数据库中案由的名称，与诉求配置.csv中的不同，这里转换为一致的。
problem_disambiguation_dict = read_json_attribute_value("LawsuitPrejudgment/lawsuit_prejudgment/api/fyt_pc_data.json", "problem_disambiguation_dict")


def _get_reasoning_graph_result(req_data: Dict):
    problem = req_data['problem']
    claim_list = req_data['claim_list']
    fact = req_data.get('fact', '')
    question_answers = req_data.get('question_answers', {})
    factor_sentence_list = req_data.get('factor_sentence_list', [])

    logging.info("=============================================================================")
    logging.info("1.problem: %s" % (problem))
    logging.info("2.claim_list: %s" % (claim_list))
    logging.info("3.fact: %s" % (fact))
    logging.info("4.question_answers: %s" % (question_answers))

    problem = problem_disambiguation_dict.get(problem, problem)
    result_dict = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)

    logging.info("5.result.result_dict: %s" % (result_dict))
    logging.info("6.service.result: %s" % (result_dict['result']))

    return {
        "status": 0,
        "error_msg": "",
        "question_asked": result_dict['question_asked'],  # 问过的问题
        "question_next": result_dict['question_next'],  # 下一个要问的问题
        "question_type": result_dict['question_type'],
        "factor_sentence_list": result_dict['factor_sentence_list'],  # 匹配到短语的列表
        "result": result_dict['result']
    }


@app.route('/reasoning_graph_result', methods=["post"])  # "service_type":'ft'
def reasoning_graph_result():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            resp_dict = _get_reasoning_graph_result(in_dict)
            return json.dumps(resp_dict, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5080, debug=False)  # , use_reloader=False)
