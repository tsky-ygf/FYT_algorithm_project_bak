import requests

t = requests.get("http://127.0.0.1:8010/get_contract_type").json()["result"]
print(t)
