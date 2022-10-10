# -*- coding: utf-8 -*-
import time
import sys
import requests
import json
"""
推理图谱即新版评估理由-接口
"""


ip = 'localhost' # test_env_ip: 192.168.1.254; waiwang: 118.31.50.13

# 调用例子1：
t1 = time.time()
problem='婚姻家庭'
claim_list=["离婚", "财产分割"] # "离婚",
fact="男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。"
question_answers={}
factor_sentence_list=[]
payload = {"problem": problem,"claim_list":claim_list,"fact":fact,"question_answers":question_answers,"factor_sentence_list":factor_sentence_list}
print("第一次输入:",payload)
r = requests.post("http://%s:5080/reasoning_graph_result"%(ip), json=payload)
result_1=r.text
print("第一次输出:",result_1)
t2 = time.time()
print(t2 - t1)


# 调用例子2：
t1 = time.time()
problem='婚姻家庭'
claim_list=['房产分割']
fact='婚后男的方父母出资首得到付，夫妻名义贷款还贷，房产证只写男方名，离婚后财产如何分配'
question_answers={'由谁付的[FF]？:您;对方;双方;您父母;双方父母;对方父母':'对方'}
factor_sentence_list=[['婚后男的方父母出资首得到付', '婚后购买', 1, ''], ['房产证只写男方名', '有房产证', 1, ''], ['房产证只写男方名', '登记在对方名下', 1, ''], ['夫妻名义贷款还贷', '首付', 1, '']]
payload = {"problem": problem,"claim_list":claim_list,"fact":fact,"question_answers":question_answers,"factor_sentence_list":factor_sentence_list}
print("第二次输入:",payload)
r = requests.post("http://%s:5080/reasoning_graph_result"%(ip), json=payload)
result_2=r.text
print("第二次输出:",result_2)
print('possibility_support', json.loads(result_2))
t2 = time.time()
print(t2 - t1)


# 调用例子3：
claim_list=["返还彩礼"]
fact="请我和老婆于2017年11月结婚，结婚时给女方彩礼30万，现要离婚，彩礼可以要回来么"
problem="婚姻家庭"
factor_sentence_list=[['彩礼可以要回来么', '一方下落不明', -1, ''], ['请我和老婆于结婚', '双方登记结婚', 1, ''], ['请我和老婆于结婚', '双方未登记结婚', -1, ''], ['结婚时给女方彩礼30万', '给付彩礼', 1, ''], ['结婚时给女方彩礼30万', '财产明确归一方所有', 1, ''], [None, '双方未共同生活', -1, ''], [None, '给付彩礼导致生活困难', -1, '']]
question_answers={'是否因给付彩礼导致生活困难？:是;否': '否', '双方是否共同生活过？:是;否': '是'}
payload = {"problem": problem,"claim_list":claim_list,"fact":fact,"question_answers":question_answers,"factor_sentence_list":factor_sentence_list}
print("第三次输入:",payload)
r = requests.post("http://%s:5080/reasoning_graph_result"%(ip), json=payload)
result_2=r.text
print("第三次输出:",result_2)
t2 = time.time()
print(t2 - t1)

