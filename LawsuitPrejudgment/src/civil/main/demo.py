# -*- coding: utf-8 -*-
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
sys.path.append(os.path.abspath('../../../../'))
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../common'))
sys.path.append(os.path.abspath('../prediction'))
from LawsuitPrejudgment.src.civil.main.reasoning_graph_predict import predict_fn
from LawsuitPrejudgment.src.civil.common.config_loader import *

logging.basicConfig(level=logging.DEBUG)


def demo():
    # 选择纠纷类型
    print('可选纠纷类型: ', ';'.join(user_ps.keys()))
    problem = input('请输入您的纠纷类型: ')

    # 选择诉求
    print('可选诉求: ', ';'.join(user_ps[problem]))
    suqius = input('请输入您的诉求: ')
    claim_list = [suqiu.strip() for suqiu in suqius.split(';') if len(suqiu.strip()) > 0]
    sentence = input('请输入遇到的问题: ')

    question_answers = {}
    repeated_question_management = {}
    result_dict = predict_fn(problem, claim_list, sentence, question_answers, {}, True, repeated_question_management)
    print('result_dict:')
    print(result_dict)
    # 诉求选择对应的默认特征
    while result_dict['question_next'] is not None:
        print('#' * 50)
        factor_sentence_list = result_dict['factor_sentence_list']
        print("特征匹配情况(根据特征表和句式关键词表):")
        print(factor_sentence_list)

        repeated_question_management = result_dict["repeated_question_management"]
        debug_info = result_dict['debug_info']
        cnt = 1
        for k, v in debug_info.items():
            print("诉求{}:\t{}".format(cnt, k))
            print(v)
            cnt += 1
        # print("当前处理的诉求:")
        # print("处理情况:")
        # # if 对诉求提问:
        # print("当前诉求的特征状态:")
        # print("当前诉求的候选路径:")
        # print("问题的来源:")
        # print("产生的问题:")
        print(result_dict['question_next'])
        answers = input('请输入答案: ')
        question_answers[result_dict['question_next']] = answers
        result_dict = predict_fn(problem, claim_list, sentence, question_answers, factor_sentence_list, True, repeated_question_management)
        print('result_dict:')
        print(result_dict)
    for suqiu, result in result_dict['result'].items():
        print('*' * 50)
        print(suqiu, result['support_or_not'], result['possibility_support'])
        print(result['reason_of_evaluation'])
        print()
        print(result['evidence_module'])
        print()
        print(result['legal_advice'])


if __name__ == '__main__':
    demo()
