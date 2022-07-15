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


@app.route('/get_civil_problem_summary', methods=["get"])
def get_civil_problem_summary():
    try:
        with open("civil_problem_summary.json") as json_data:
            problem_summary = json.load(json_data)["value"]
        return json.dumps({"success": True, "error_msg": "", "value": problem_summary}, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"success": False, "error_msg": repr(e), "value": None}, ensure_ascii=False)


@app.route('/reasoning_graph_result', methods=["post"])  # "service_type":'ft'
def reasoning_graph_result():
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
                                "similar_rate": "88%",
                                "title": "原告王某某与被告郝某某等三人婚约财产纠纷一等婚约财产纠纷一审民事判决书",
                                "court": "公主岭市人民法院",
                                "judge_date": "2016-04-11",
                                "case_number": "（2016）吉0381民初315号",
                                "tag": "彩礼 证据 结婚 给付 协议 女方 当事人 登记 离婚",
                                "win_or_not": True
                            },
                            {
                                "doc_id": "ws_c4b1e568-b253-4ac3-afd7-437941f1b17a",
                                "similar_rate": "80%",
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5090, debug=True)  # , use_reloader=False)
