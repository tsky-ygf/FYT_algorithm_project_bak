#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/25 10:15
# @Author  : Adolf
# @Site    : 
# @File    : keyword_feature_matching.py
# @Software: PyCharm
import requests

ip = '172.19.82.199'
port = 9500

def get_match_result(sentence, problem, suqiu):
    request_data = {
        "sentence": sentence,
        "problem": problem,
        "suqiu": suqiu
    }

    r = requests.post("http://%s:%s/keyword_feature_matching" % (ip, port), json=request_data)
    result = r.json()
    return result

# Use Cases
# 用例1：匹配正确
sentence = '2014年6月，我借给了何三宇、冯群华20000元并写了借条，约定月息3%，在2014年10月14日前一次还清，同时谭学民、蔡金花作了担保人。到期后，何三宇、冯群华迟迟不还款，现在我想让他们按照约定，还我本金及利息。'
problem="借贷纠纷"
suqiu="民间借贷"
print("用例1 输出:")
print(get_match_result(sentence, problem, suqiu))

# 用例2：匹配错误
sentence = '2015年7月17日，被告向我借款10万元，并出具欠条一张，约定借款期限为3个月，并口头约定利息按照同期银行贷款利率计算，借款到期后，多次催要无果，要求被告偿还借款及利息。'
problem="借贷纠纷"
suqiu="民间借贷"
print("用例2 输出:")
print(get_match_result(sentence, problem, suqiu))

# 用例3：未匹配到或异常
sentence = '2015年7月17日，被告向我借款10万元，并出具欠条一张，约定借款期限为3个月，并口头约定利息按照同期银行贷款利率计算，借款到期后，多次催要无果，要求被告偿还借款及利息。'
problem="借贷纠纷"
suqiu="错误的诉求名称"
print("用例3 输出:")
print(get_match_result(sentence, problem, suqiu))