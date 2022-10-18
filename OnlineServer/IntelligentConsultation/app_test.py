#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : app_test.py
# @Software: PyCharm
from pip import main
import requests

URL = "http://127.0.0.1:8130"


def test_get_query_answer_with_source():
    # 准备测试数据
    body = {
        "question": "公司交不起税怎么办",
        "query_source": "专题"
        # "source_end": "personal"
    }

    # 执行被测试程序
    resp_json = requests.post(URL + "/get_query_answer_with_source", json=body).json()
    answer = resp_json.get("answer")

    print(answer)
    # 验证测试条件
    # assert answer == """依法纳税是每个企业和公民的基本义务，如：\n
    # 1.纳税人因有特殊困难，不能按期缴纳税款的，经省、自治区、直辖市国家税务局、地方税务局批准，可以延期缴纳税款，但是最长不得超过三个月。\n
    # 2.纳税人交不出税，遇到自然灾害或者生产经营恶化时可以向税务部门申请减免税务或申请延期分批缴税；当无法经营下去时可以申请破产，清算抵税。\n
    # 3.对欠缴税款的，税务机关有权进行催缴，同时按日加收滞纳税款万分之五的滞纳金。纳税人欠缴应纳税款，采取转移或者隐匿财产的手段，妨碍税务机关
    # 追缴欠缴的税款的，由税务机关追缴欠缴的税款、滞纳金，并处欠缴税款百分之五十以上五倍以下的罚款；构成犯罪的，依法追究刑事责任。纳税人欠缴应
    # 纳税款，采取转移或者隐匿财产的手段，致使税务机关无法追缴欠缴的税款，数额在一万元以上不满十万元的，处三年以下有期徒刑或者拘役，并处或者
    # 单处欠缴税款一倍以上五倍以下罚金;数额在十万元以上的，处三年以上七年以下有期徒刑，并处欠缴税款一倍以上五倍以下罚金。"""


if __name__ == '__main__':
    test_get_query_answer_with_source()
