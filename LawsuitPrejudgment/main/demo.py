# -*- coding: utf-8 -*-
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../common'))
from reasoning_graph_predict import predict_fn
from config_loader import *
logging.basicConfig(level=logging.DEBUG)
def demo():
    # 选择纠纷类型
    print('可选纠纷类型: ', ';'.join(user_ps.keys()))
    problem = input('请输入您的纠纷类型: ')
    # 选择诉求
    print('可选诉求: ', ';'.join(user_ps[problem]))
    suqius = input('请输入您的诉求: ')
    claim_list = [suqiu.strip() for suqiu in suqius.split(';') if len(suqiu.strip())>0]
    sentence = input('请输入遇到的问题: ')

    question_answers = {}
    result_dict = predict_fn(problem, claim_list, sentence, question_answers, {}, True)
    print(result_dict)
    # 诉求选择对应的默认特征
    while result_dict['question_next'] is not None:
        factor_sentence_list = result_dict['factor_sentence_list']
        print(factor_sentence_list)
        print(result_dict['question_next'])
        answers = input('请输入答案: ')
        question_answers[result_dict['question_next']] = answers
        result_dict = predict_fn(problem, claim_list, sentence, question_answers, factor_sentence_list, True)
    for suqiu, result in result_dict['result'].items():
        print('*'*50)
        print(suqiu, result['support_or_not'], result['possibility_support'])
        print(result['reason_of_evaluation'])
        print()
        print(result['evidence_module'])
        print()
        print(result['legal_advice'])


if __name__ == '__main__':
    demo()
