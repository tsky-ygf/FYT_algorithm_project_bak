import requests


def test_get_administrative_situation_list():
    url = "http://101.69.229.138:8100/get_situation"
    data = {
        "administrative_type": "tax"
    }
    resp = requests.post(url, json=data)
    assert resp.json().get("success") is True
    pass


def test_get_administrative_result():
    url = "http://101.69.229.138:8100/get_administrative_result"
    data = {
        "administrative_type": "tax",
        "situation": "逃避税务机关检查"
    }
    resp = requests.post(url, json=data)
    assert resp.json().get("success") is True
    pass