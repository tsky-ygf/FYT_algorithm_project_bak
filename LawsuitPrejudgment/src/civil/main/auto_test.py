# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../common'))
from common import multi_processing_data, single_case_match, LogicTree, suqiu_factor_dict, problem_bkw_dict, suqiu_bkw_dict, problem_suqius


def test(problem, num, process_num):
    raw = pd.read_csv('../data/prediction/' + problem + '_train.csv', encoding='utf-8')
    raw = raw.dropna().sample(frac=1)
    result = []
    for suqiu in problem_suqius[problem]:
        data = raw[raw['new_suqiu']==suqiu][:num].copy()
        data_x_kw = multi_processing_data((data['chaming_fact'] + '。' + data['renwei_fact']).values,
                                          min(process_num, len(data)),
                                          problem, None)
        print("data_x_kw: ", data_x_kw.shape)
        for i, f in enumerate(problem_bkw_dict[problem].index):
            data['factor:' + problem + '_' + f] = data_x_kw[:, i]
        data['positive_factor'] = data.apply(lambda row: [c.split('_')[1] for c in data.columns[row==1] if c.startswith('factor:')], axis=1)
        data['negative_factor'] = data.apply(lambda row: [c.split('_')[1] for c in data.columns[row==-1] if c.startswith('factor:')], axis=1)

        for index, row in data.iterrows():
            matched_factor = []
            suqiu = row['new_suqiu']
            tree = LogicTree(problem, suqiu)
            if problem + '_' + suqiu in suqiu_factor_dict:
                for factor, flag in suqiu_factor_dict[problem + '_' + suqiu].items():
                    tree.add_match_result(factor, flag, None)
                    matched_factor.append(factor + ':' + str(flag))
            for factor in row['positive_factor']:
                tree.add_match_result(factor, 1, None)
                matched_factor.append(factor + ':1')
            for factor in row['negative_factor']:
                tree.add_match_result(factor, -1, None)
                matched_factor.append(factor + ':-1')

            question = tree.get_next_question()
            if question is not None:
                result.append([problem, suqiu, row['chaming_fact'], row['renwei_fact'], row['label'], '\n'.join(matched_factor), '特征匹配不全:'+question])
            else:
                r = tree.print_logic_result(0.4)
                if r==1 and row['label']==0:
                    result.append([problem, suqiu, row['chaming_fact'], row['renwei_fact'], row['label'], '\n'.join(matched_factor), '路径支持文书不支持'])
                if r==-1 and row['label']==1:
                    result.append([problem, suqiu, row['chaming_fact'], row['renwei_fact'], row['label'], '\n'.join(matched_factor), '路径不支持文书支持'])
    return result


def factor_test(problem, num):
    raw = pd.read_csv('../data/prediction/' + problem + '_train.csv', encoding='utf-8')
    raw = raw.dropna().sample(frac=1)
    result = []
    for suqiu in problem_suqius[problem]:
        data = raw[raw['new_suqiu']==suqiu][:num].copy()
        for index, row in data.iterrows():
            sentence_factor = single_case_match(row['chaming_fact'], problem, suqiu)
            positive_factors = []
            negative_factors = []
            unmatched_factors = []
            for factor in suqiu_bkw_dict[problem + '_' + suqiu].index:
                if factor not in sentence_factor:
                    unmatched_factors.append(factor)
                elif sentence_factor[factor][1]==1:
                    positive_factors.append(factor+':'+sentence_factor[factor][0])
                else:
                    negative_factors.append(factor+':'+sentence_factor[factor][0])
            result.append([problem, suqiu, row['chaming_fact'], '\n'.join(positive_factors), '\n'.join(negative_factors), '\n'.join(unmatched_factors)])
    return result


if __name__=='__main__':
    # result = test('劳动社保', 20, 3)
    # result = pd.DataFrame(result, columns=['problem', 'suqiu', 'chaming', 'renwei', 'panjue', 'matched_factors', 'result'])
    # result.to_csv('../data/auto_test_result.csv', index=False, encoding='utf-8')

    result = factor_test('劳动社保', 5)
    result = pd.DataFrame(result, columns=['problem', 'suqiu', 'chaming', 'positive_factors', 'negative_factors', 'unmatched_factors'])
    result.to_csv('../data/factor_test_result.csv', index=False, encoding='utf-8')
