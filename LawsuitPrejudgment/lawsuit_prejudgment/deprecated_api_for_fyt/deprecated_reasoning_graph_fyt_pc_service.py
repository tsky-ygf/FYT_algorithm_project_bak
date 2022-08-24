# -*- coding: utf-8 -*-
import json
import traceback
import logging
import logging.handlers
from flask import Flask
from flask import request
from LawsuitPrejudgment.Administrative.administrative_api_v1 import get_administrative_prejudgment_situation

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
    """解析请求数据并以json形式返回"""
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5080, debug=False)  # , use_reloader=False)
