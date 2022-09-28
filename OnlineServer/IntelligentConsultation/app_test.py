#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
import requests

URL = "http://127.0.0.1:8130"

def test_get_query_answer():
    # 准备测试数据
    body = {
        "question": "争议解决",
        "source_end": "personal"
    }

    # 执行被测试程序
    resp_json = requests.post(URL+"/get_query_answer", json=body).json()
    answer = resp_json.get("answer")

    # 验证测试条件
    assert answer == "【争议解决】\n1.双方发生合同纠纷，可以先协商解决，也可以向当地人民调解委员会申请调解；若协商不成，约定了争议解决条款的，也可以根据合同约定进行仲裁或诉讼；若合同没有约定争议解决条款，可以向合同履行地、被告住所地人民法院起诉。\n2.双方发生劳动纠纷，优先协商解决，也可以请工会或者第三方共同与用人单位协商；若协商不成，可以向劳动合同履行地或者用人单位所在地的劳动争议仲裁委员会申请仲裁，对仲裁结果不服的，可以向人民法院起诉。"
    pass
