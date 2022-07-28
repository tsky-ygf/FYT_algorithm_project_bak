# -*- coding: utf-8 -*-
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
from collections import OrderedDict

from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn
from LawsuitPrejudgment.common.config_loader import *
"""
# 交互的获取用户输入，保存问题、用户输入、预测结果
"""
des_folder = '../data/test_cases_20220217'

def demo():
    logging.basicConfig(level=logging.ERROR)
    # 记录输入输出
    json_data = OrderedDict()
    # 选择纠纷类型
    print('=' * 80)
    doc_id = input('请输入测试序号: ')
    print('可选纠纷类型: ', ';'.join(user_ps.keys()))
    problem = input('请输入您的纠纷类型: ')
    if problem == '000':    # 退出测试
        return False
    # 选择诉求
    print('可选诉求: ', ';'.join(user_ps[problem]))
    suqius = input('请输入您的诉求: ')
    claim_list = [suqiu.strip() for suqiu in suqius.split(';') if len(suqiu.strip())>0]
    sentence = input('请输入遇到的问题: ')
    json_data.update({'problem': problem, 'suqius': suqius, 'sentence': sentence, 'question_answers': [], 'result': []})

    question_answers = {}
    result_dict = predict_fn(problem, claim_list, sentence, question_answers, {}, True)
    print(result_dict)
    # 诉求选择对应的默认特征
    while result_dict['question_next'] is not None:
        factor_sentence_list = result_dict['factor_sentence_list']
        # print(factor_sentence_list)
        print(result_dict['question_next'])
        answers = input('请输入答案: ')
        question_answers[result_dict['question_next']] = answers
        json_data['question_answers'].append({'question': result_dict['question_next'], 'answers': answers})
        result_dict = predict_fn(problem, claim_list, sentence, question_answers, factor_sentence_list, True)
    for suqiu, result in sorted(result_dict['result'].items()):
        print('*'*50)
        print(suqiu, result['support_or_not'], result['possibility_support'])
        print(result['reason_of_evaluation'])
        print()
        print(result['evidence_module'])
        print()
        print(result['legal_advice'])
        answers = input('请输入真实判决结果0/1：')
        print('真实判决结果为:', answers)
        json_data['result'].append({'suqiu': suqiu, 'support_or_not': result['support_or_not'],
                                    'possibility_support': result['possibility_support'],
                                    'reason_of_evaluation': result['reason_of_evaluation'],
                                    'true_label': answers})


    json_data['result_code'] = []
    for e in json_data['result']:
        json_data['result_code'].append('1' if e['support_or_not'] == '支持' else '0')
    json_data['result_code'] = '_'.join(json_data['result_code'])
    json_data['label_code'] = '_'.join([e['true_label'] for e in json_data['result']])
    json_data['doc_id'] = doc_id
    num = max([int(e.split('.')[0]) for e in os.listdir(des_folder) if not e.startswith('all')] + [-1]) + 1
    json.dump(json_data, open(os.path.join(des_folder, str(num)+'.json'), 'w'), ensure_ascii=False)
    return True



if __name__ == '__main__':
    flag = True
    while flag:
        flag = demo()

