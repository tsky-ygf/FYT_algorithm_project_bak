import time
import sys
import requests

t1 = time.time()
fact = "借钱不还"
payload = {"problem": "民间借贷纠纷", "claim_list": [], "fact": fact}
r = requests.post(
    "http://172.19.82.199:8163/top_k_similar_narrative", json=payload
)  # 47.99.90.181
t2 = time.time()

print(r.text)
print(t2 - t1)
