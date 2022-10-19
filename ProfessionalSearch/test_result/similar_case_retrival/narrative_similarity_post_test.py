import json
import time
import sys
import requests

from Utils import print_run_time


@print_run_time
def similar_case_bug():
    fact = "借钱不还"
    payload = {"problem": "民间借贷纠纷", "claim_list": [], "fact": fact}
    r = requests.post(
        "http://127.0.0.1:8132/top_k_similar_narrative", json=payload
    ).json()  # 47.99.90.181
    print(type(r))
    print(json.loads(r)["dids"])


if __name__ == '__main__':
    similar_case_bug()
