#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 13:50
# @Author  : Adolf
# @Site    : d j
# @File    : use_app.py
# @Software: PyCharm
import json
import os.path
import re
import time
from pprint import pprint

import requests

# url = "http://101.69.229.138:8111/upload_docx_to_get_text"
file_path = '/home/fyt/data/DocData/jietiao/jietiao2.docx'
f = open(file_path, 'rb')
print("type ",type(f))
f.close()
print(type("213"))
# print(os.path.exists(file_path))
# r = requests.post(url, files={'file':open(file_path,'rb')})
# print(r.text)


# url = "http://101.69.229.138:8111/file_link_path_to_text"
# file_path = 'https://nblh-fyt.oss-cn-hangzhou.aliyuncs.com/fyt/20220916/55712bdc-694a-438f-bcb0-1f5b66dd9bb5.docx'
# inputs = {'file_path':file_path}
# r = requests.post(url,json=inputs)
# print(r.text)

# os.system('wget https://nblh-fyt.oss-cn-hangzhou.aliyuncs.com/fyt/20220916/55712bdc-694a-438f-bcb0-1f5b66dd9bb5.docx')

# url = "http://172.19.82.199:8110/get_contract_review_result"
# req_data = {
#     "contract_type_id": "fangwuzulin",
#     "user_standpoint_id": "party_a",
#     "contract_content": """"""
# }
# start_time = time.time()
# resp_json = requests.post(url, json=req_data).json()
# print(resp_json)
#
# time_cost = time.time() - start_time
# print("time cost:{}".format(time_cost))
