import requests


def test_get_administrative_situation_list():
    url = "http://101.69.229.138:8100/get_administrative_problem_and_situation_by_type_id?type_id={}".format("tax")
    resp_json = requests.get(url).json()
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("result")) > 0
    pass


def test_get_administrative_result():
    url = "http://101.69.229.138:8100/get_administrative_result"
    data = {
        "type_id": "tax",
        "situation": "逃避税务机关检查"
    }
    resp_json = requests.post(url, json=data).json()
    assert resp_json is not None
    assert resp_json.get("success") is True
    assert len(resp_json.get("result")) > 0
    pass
