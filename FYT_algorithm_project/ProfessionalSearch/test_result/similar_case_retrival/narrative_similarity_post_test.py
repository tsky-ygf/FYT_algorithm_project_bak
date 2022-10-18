import time
import sys
import requests

from Utils import print_run_time


@print_run_time
def similar_case_bug():
    fact = "借钱不还"
    payload = {"problem": "民间借贷纠纷", "claim_list": [], "fact": fact}
    r = requests.post(
        "http://172.19.82.199:8163/top_k_similar_narrative", json=payload
    )  # 47.99.90.181
    print(r.text)


if __name__ == '__main__':
    similar_case_bug()
