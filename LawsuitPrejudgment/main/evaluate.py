# -*- coding: utf-8 -*-
import pandas as pd

"""
特征匹配率：匹配到n个特征的案例数量/总案例数量——用户信息的利用程度
特征匹配数量: mean(匹配到的特征数量)——用户信息的利用程度

平均提问次数：总提问次数/总案例数量——问题是否太多
问题多样性：存在相同问题的案例对数量/总案例对数量——是否总是问相同问题
最高问题频率：max(问题提问次数/总案例数量)——是否总是问相同问题

评估理由逻辑性：有逻辑的案例数量/总案例数量
"""


# data = pd.read_csv('../data/test.csv',encoding='utf-8')
# data = data[data['suqiu'].isin(problem_suqius['婚姻家庭'])]
#
# num = []
# for index, row in data.iterrows():
#     sentence_factor = single_case_match(row['content'], row['problem'], row['suqiu'])
#     num.append(len(sentence_factor))
# num = np.array(num)
# print('特征匹配率: %s, %s, %s'%(sum(num==1)/len(num),sum(num==2)/len(num),sum(num==3)/len(num)))
# print('特征匹配数量: %s' % (num.mean()))

data = pd.read_csv('../data/测试结果.csv', encoding='utf-8')
print('旧方案提问比例%s'%(len(data[data['question_old']=='-1'])/len(data)))
print('新方案提问比例%s'%(len(data[data['question_new']=='-1'])/len(data)))
print('旧方案提问种类%s'%(len(data['question_old'].drop_duplicates())))
print('新方案提问种类%s'%(len(data['question_new'].drop_duplicates())))
print('旧方案评估理由得分%s'%(data['reason_old'].mean()))
print('新方案评估理由得分%s'%(data['reason_new'].mean()))

question_values = [data['question_old'].values, data['question_new'].values]
for values in question_values:
    ask_num = 0
    repeat_num = 0
    different_num = 0
    total_num = 0
    for i in range(len(values)):
        set_a = set([v.strip() for v in values[i].split('，')])
        if values[i]=='-1':
            continue
        ask_num += len(set_a)
        for j in range(i+1, len(data)):
            if values[j]=='-1':
                continue
            total_num += 1
            set_b = set([v.strip() for v in values[j].split('，')])
            if len(set_a & set_b)>0:
                repeat_num += 1
            if len(set_a | set_b)>len(set_a):
                different_num += 1
    print('平均提问次数%s'%(ask_num/len(values)))
    print('有相同问题比例%s'%(repeat_num/total_num))
    print('有不同问题比例%s'%(different_num/total_num))
