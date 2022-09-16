#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 13:50
# @Author  : Adolf
# @Site    : d j
# @File    : test_app.py
# @Software: PyCharm
import json
import os.path
from pprint import pprint

import requests

# url = "http://101.69.229.138:8111/upload_docx_to_get_text"
# file_path = '/home/fyt/data/DocData/jietiao/jietiao2.docx'
# print(os.path.exists(file_path))
# r = requests.post(url, files={'file':open(file_path,'rb')})
# print(r.text)


url = "http://101.69.229.138:8111/file_link_path_to_text"
file_path = 'https://nblh-fyt.oss-cn-hangzhou.aliyuncs.com/fyt/20220916/55712bdc-694a-438f-bcb0-1f5b66dd9bb5.docx'
inputs = {'file_path':file_path}
r = requests.post(url,json=inputs)
print(r.text)

# os.system('wget https://nblh-fyt.oss-cn-hangzhou.aliyuncs.com/fyt/20220916/55712bdc-694a-438f-bcb0-1f5b66dd9bb5.docx')


