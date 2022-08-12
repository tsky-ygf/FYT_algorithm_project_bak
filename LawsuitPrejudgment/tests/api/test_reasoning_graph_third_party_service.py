import requests

from LawsuitPrejudgment.lawsuit_prejudgment.api.reasoning_graph_third_party_service import \
    _get_supported_administrative_types


def test_get_civil_problem_summary():
    url = "http://101.69.229.138:8100/get_civil_problem_summary"
    resp_json = requests.get(url).json()
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("value")) > 0
    pass


def test_get_administrative_type():
    url = "http://101.69.229.138:8100/get_administrative_type"
    resp_json = requests.get(url).json()

    print(resp_json)
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("result")) > 0
    assert resp_json["result"][0].get("type_id")
    assert resp_json["result"][0].get("type_name")


def test_get_administrative_situation_list():
    supported_administrative_types = _get_supported_administrative_types()
    type_id_list = [item["type_id"] for item in supported_administrative_types]

    for type_id in type_id_list:
        url = "http://101.69.229.138:8100/get_administrative_problem_and_situation_by_type_id?type_id={}".format(type_id)
        resp_json = requests.get(url).json()
        assert resp_json is not None, type_id
        assert resp_json.get("success") is True, type_id
        assert len(resp_json.get("result")) > 0, type_id
    pass


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
    pass


def test_get_criminal_result():
    url = "http://101.69.229.138:8100/get_criminal_result"
    data = {
        "question": "我打人了，怎么办"
    }
    resp_json = requests.post(url, json=data).json()
    print(resp_json)
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("result")) > 0
    pass
