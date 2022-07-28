# -*- coding: utf-8 -*-
import json
import logging
import os

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from collections import OrderedDict
import sys

from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn
from LawsuitPrejudgment.common.config_loader import *

"""
# 从保存的文件中获取用户输入，如果问题路径有变化，记录空，否则记录新的预测结果
"""


def demo(folder):
    logging.basicConfig(level=logging.ERROR)
    file_path = os.path.join(folder, 'all_20220212.xlsx')
    df = pd.read_excel(file_path)
    df['new_result'] = ''
    df['new_result_code'] = ''
    df['new_label_code'] = ''
    df['new_same_path'] = '0'
    df['new_factor_sentence_list'] = ''
    column_num = df.columns.get_loc("new_result")
    for i, row in df.iterrows():
        # 记录新的输出
        new_result = []
        problem, suqius, sentence = row['problem'], row['suqius'], row['sentence']
        # 选择纠纷类型
        # 选择诉求
        claim_list = [suqiu.strip() for suqiu in suqius.split(';') if len(suqiu.strip()) > 0]

        question_answers = {}
        result_dict = predict_fn(problem, claim_list, sentence, question_answers, {}, True)
        print('*' * 50)
        print(result_dict)
        # 对于记录中的每一个问题和用户输入
        empty_question_flag = False
        if result_dict['question_next'] is None:  # 匹配的特征构成至少一个路径，问题为空
            empty_question_flag = True
        flag = True
        for qa in eval(row['question_answers']):
            factor_sentence_list = result_dict['factor_sentence_list']
            if result_dict['question_next'] == qa['question']:
                # 问题一样
                question_answers[result_dict['question_next']] = qa['answers']
                result_dict = predict_fn(problem, claim_list, sentence, question_answers, factor_sentence_list, True)
            else:
                flag = False
                break

        if empty_question_flag or (flag and result_dict['question_next'] is None):  # 路径一样，且没有后续问题
            old_result = {e['suqiu']: e['true_label'] for e in eval(row['result'])}
            for suqiu, result in result_dict['result'].items():
                print(suqiu, result['support_or_not'], result['possibility_support'])
                if suqiu in old_result:
                    new_result.append({'suqiu': suqiu, 'support_or_not': result['support_or_not'],
                                       'reason_of_evaluation': result['reason_of_evaluation'],
                                       'possibility_support': result['possibility_support'],
                                       'true_label': old_result[suqiu]})
                else:
                    break

        result_code = []
        for e in new_result:
            result_code.append('1' if e['support_or_not'] == '支持' else '0')
        result_code = '_'.join(result_code)
        label_code = '_'.join([e['true_label'] for e in new_result])

        df.iloc[i, column_num] = json.dumps(new_result, ensure_ascii=False)
        df.iloc[i, column_num + 1] = result_code
        df.iloc[i, column_num + 2] = label_code
        if flag and result_dict['question_next'] is None:
            df.iloc[i, column_num + 3] = '1'
        df.iloc[i, column_num + 4] = json.dumps(result_dict['factor_sentence_list'], ensure_ascii=False)

    def get_new_support_rate(x):
        return '1' if x['new_label_code'] and x['new_result_code'] == x['new_label_code'] else '0'

    df['new_支持准确率'] = df.apply(get_new_support_rate, axis=1)
    df.to_excel(os.path.join(folder, '租赁合同测试集100_20220217.xlsx'), index=False)


if __name__ == '__main__':
    des_folder = '../data/test_cases_20220210'
    demo(des_folder)
