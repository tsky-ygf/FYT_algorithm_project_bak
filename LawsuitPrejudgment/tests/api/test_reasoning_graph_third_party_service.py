import logging

import requests

# from LawsuitPrejudgment.lawsuit_prejudgment.api.reasoning_graph_third_party_service import \
#     _get_supported_administrative_types

attributes_in_similar_case = {"doc_id", "similar_rate", "title", "court", "judge_date", "case_number", "tag", "is_guiding_case"}
attributes_in_applicable_law = {"law_id", "law_name", "law_item", "law_content"}
attributes_in_judging_rule = {"rule_id", "content", "source", "source_url"}


def test_get_civil_problem_summary():
    url = "http://101.69.229.138:8100/get_civil_problem_summary"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("value")) > 0
    pass


def test_get_template_by_problem_id():
    url = "http://101.69.229.138:8100/get_template_by_problem_id"
    params = {"problem_id": 1564}
    resp_json = requests.get(url, params=params).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("value")
    assert resp_json["value"]["template"] == "x年x月x日，x与x签订运输合同/搭乘车辆，x履行/未履行运输义务，x支付/未支付货款，或x存在x违约行为，造成x损害后果。（请具体描述过程及结果，如突然刹车，与客车相撞等）"


def test_get_claim_list_by_problem_id():
    url = "http://101.69.229.138:8100/get_claim_list_by_problem_id"
    params = {"problem_id": 1536}
    resp_json = requests.get(url, params=params).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("value")
    assert next((item.get("claim") for item in resp_json["value"] if item.get("claim") == "请求返还彩礼"), None)


def test_should_ask_next_question_when_reasoning_graph_result():
    url = "http://101.69.229.138:8100/reasoning_graph_result"
    body = {
        "problem": "婚姻家庭",
        "claim_list": ["请求离婚"],
        "fact": "男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。",
        "question_answers": {},
        "factor_sentence_list": []
    }
    resp_json = requests.post(url, json=body).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("question_next")
    assert resp_json.get("result") is None


def test_show_have_report_when_reasoning_graph_result():
    url = "http://101.69.229.138:8100/reasoning_graph_result"
    body = {
        "problem": "婚姻家庭",
        "claim_list": ["请求离婚"],
        "fact": "男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。",
        "question_answers": {
            "共同生活时间是否很短？:是;否": "是"
        },
        "factor_sentence_list": [["男女双方自愿/不自愿（不自愿的原因）登记结婚", "双方自愿离婚", -1, ""], ["男女双方自愿/不自愿（不自愿的原因）登记结婚", "双方非自愿结婚", 1, ""], ["男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚", "一方重婚", -1, ""]]
    }
    resp_json = requests.post(url, json=body).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success")
    assert resp_json.get("question_next") is None
    assert resp_json.get("result")


def test_get_administrative_type():
    url = "http://101.69.229.138:8100/get_administrative_type"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("result")) > 0
    assert resp_json["result"][0].get("type_id")
    assert resp_json["result"][0].get("type_name")


# def test_get_administrative_situation_list():
#     supported_administrative_types = _get_supported_administrative_types()
#     type_id_list = [item["type_id"] for item in supported_administrative_types]
#
#     for type_id in type_id_list:
#         url = "http://101.69.229.138:8100/get_administrative_problem_and_situation_by_type_id?type_id={}".format(type_id)
#         resp_json = requests.get(url).json()
#         assert resp_json is not None, type_id
#         assert resp_json.get("success") is True, type_id
#         assert len(resp_json.get("result")) > 0, type_id
#     pass


def test_get_administrative_result():
    url = "http://101.69.229.138:8100/get_administrative_result"
    data = {
        "type_id": "tax",
        "situation": "逃避税务机关检查"
    }
    resp_json = requests.post(url, json=data).json()
    print(resp_json)
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("result")) > 0
    assert set(resp_json["result"]["similar_case"][0].keys()) == attributes_in_similar_case
    assert set(resp_json["result"]["applicable_law"][0].keys()) == attributes_in_applicable_law
    assert set(resp_json["result"]["judging_rule"][0].keys()) == attributes_in_judging_rule
    pass


def test_get_criminal_result():
    url = "http://101.69.229.138:8100/get_criminal_result"
    data = {
        "fact": "我打人了，怎么办？",
        "question_answers": {},
        "factor_sentence_list": []
    }

    resp_json = requests.post(url, json=data).json()
    print(resp_json)

    assert resp_json is not None
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") is None
    assert len(resp_json.get("result")) > 0
    assert set(resp_json["result"]["applicable_law"][0].keys()) == attributes_in_applicable_law
    assert set(resp_json["result"]["similar_case"][0].keys()) == attributes_in_similar_case
    assert set(resp_json["result"]["judging_rule"][0].keys()) == attributes_in_judging_rule
    pass


def test_should_ask_question_when_get_criminal_result():
    url = "http://101.69.229.138:8100/get_criminal_result"
    data = {
        "fact": "2020年7、8月份的一天，小黄电话联系我要买一小包毒品，我们约好当天下午3点在杭州市郊区某小区附近碰头。当天下午我们碰头后，我将一小包毒品塞给了小黄，收了他1500元，然后我们就各自回去了。",
        "question_answers": {},
        "factor_sentence_list": []
    }
    resp_json = requests.post(url, json=data).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") == "请问贩卖的毒品是以下哪种类型？:冰毒;海洛因;鸦片;其他"
    assert resp_json.get("question_type") == "1"
    assert resp_json.get("result") is None

    data = {
        "fact": "2020年7、8月份的一天，小黄电话联系我要买一小包毒品，我们约好当天下午3点在杭州市郊区某小区附近碰头。当天下午我们碰头后，我将一小包毒品塞给了小黄，收了他1500元，然后我们就各自回去了。",
        "question_answers": {
            "请问贩卖的毒品是以下哪种类型？:冰毒;海洛因;鸦片;其他": "冰毒"
        },
        "factor_sentence_list": []
    }
    resp_json = requests.post(url, json=data).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") == "请问贩卖的毒品数量有多少克？"
    assert resp_json.get("question_type") == "0"
    assert resp_json.get("result") is None


def test_should_get_report_when_get_criminal_result():
    url = "http://101.69.229.138:8100/get_criminal_result"
    data = {
        "fact": "2020年7、8月份的一天，小黄电话联系我要买一小包毒品，我们约好当天下午3点在杭州市郊区某小区附近碰头。当天下午我们碰头后，我将一小包毒品塞给了小黄，收了他1500元，然后我们就各自回去了。",
        "question_answers": {
            "请问贩卖的毒品是以下哪种类型？:冰毒;海洛因;鸦片;其他": "冰毒",
            "请问贩卖的毒品数量有多少克？": 1
        },
        "factor_sentence_list": []
    }
    resp_json = requests.post(url, json=data).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") is None
    assert resp_json.get("question_type") == "1"
    assert resp_json.get("result")

    data = {
        "fact": "我打人了，怎么办？",
    }
    resp_json = requests.post(url, json=data).json()

    print(resp_json)
    assert resp_json
    assert resp_json.get("success") is True
    assert resp_json.get("question_next") is None
    assert resp_json.get("question_type") == "1"
    assert resp_json.get("result")