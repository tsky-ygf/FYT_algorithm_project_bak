# -*- coding: utf-8 -*-
import json
import traceback
import logging
import logging.handlers
from flask import Flask
from flask import request
import sys
import os
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../common'))
sys.path.append(os.path.abspath('../prediction'))
from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn
from LawsuitPrejudgment.common.config_loader import *
"""
推理图谱的接口，添加了中间过程信息，用于测试。
"""
app = Flask(__name__)
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
logger.setLevel(logging.INFO)
handler = logging.handlers.TimedRotatingFileHandler('./testing_service.log', when='D', interval=1)
handler.setFormatter(formatter)
logger.addHandler(handler)

# handler2 = logging.StreamHandler()  # StreamHandler是输出到控制台
# logger.addHandler(handler2)

# 接口的格式
# {“question_asked”: { < 问题和候选答案字符串串 >: < ⽤用户选择的答案 >},
# “question_next”: ‘ < 问题和候选答案字符串串 >’,
# “factor_sentence_list": [ [“ < ⽤用户输⼊入匹配到的短句句 > ", " < 特征名 > ", <整数>, “<特征对应的正则>”]],
# "result": {“诉求1": {
# “reason_of_evaluation”: ’ < 评估理理由 >’,
# "evidence_module": ‘ < 证据模块 >’,
# “legal_advice: '<法律律建议>',
# "possibility_support": < 0到1之间的数字 >,
# “support_or_not: '<⽀支持或不不⽀支持>'
# }}
# "error_msg": "",
# "status": 0
# }


@app.route('/reasoning_graph_testing/get_result', methods=["post"])  # "service_type":'ft'
def hello_world():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            problem = in_dict['problem']
            claim_list = in_dict['claim_list']
            fact = in_dict.get('fact', '')
            question_answers = in_dict.get('question_answers', {})
            factor_sentence_list = in_dict.get('factor_sentence_list', [])

            logging.info("=============================================================================")
            logging.info("1.problem: %s" % (problem))
            logging.info("2.claim_list: %s" % (claim_list))
            logging.info("3.fact: %s" % (fact))
            logging.info("4.question_answers: %s" % (question_answers))

            result_dict = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list, True)

            logging.info("5.result.result_dict: %s" % (result_dict))

            question_asked = result_dict['question_asked']   # 问过的问题
            question_next = result_dict['question_next']  # 下一个要问的问题
            question_type = result_dict['question_type']
            factor_sentence_list = result_dict['factor_sentence_list']   # 匹配到短语的列表
            debug_info = result_dict['debug_info']
            result = result_dict['result']
            logging.info("6.service.result: %s" % (result))
            return json.dumps({'question_asked': question_asked, 'question_next': question_next, "question_type": question_type, "factor_sentence_list": factor_sentence_list, "debug_info": debug_info, "result": result, "error_msg": "", "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)
    except Exception as e:
       logging.info(traceback.format_exc())
       return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


@app.route('/reasoning_graph_testing/get_anyou_list', methods=["get"])
def get_anyou_list():
    return {
        "anyou_list": user_ps.keys().tolist()
    }


@app.route('/reasoning_graph_testing/get_suqiu_list', methods=["get"])
def get_suqiu_list():
    anyou = request.args.get("anyou")
    return {
        "suqiu_list": user_ps[anyou]
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5099, debug=True)  # , use_reloader=False)
