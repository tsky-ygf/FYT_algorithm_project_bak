# -*- coding: utf-8 -*-
import pandas as pd
import random
import re
import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../common'))
from data_prepare import sentence_process
from ner_predict import predict
from common import multi_processing_data, problem_bkw_dict, suqiu_bkw_dict, logic_ps


max_seq_length = 64


def proof_extract(problem_list, num_per_factor=5, process_num=30):
    """
    提取证据并增加匹配到的特征
    :param num_per_factor: 每个特征的案件数量
    :param process_num: 进程数
    """

    # # 读取原始数据
    # raw_data = []
    # for root, dirs, files in os.walk('/datadisk2/tyin/20190328/'):
    #     for file in files:
    #         print(os.path.join(root, file))
    #         if len(re.findall('|'.join(['劳动纠纷', '社保纠纷', '工伤赔偿', '提供劳务者受害责任纠纷', '提供劳务者致害责任纠纷']), file))==0:
    #             continue
    #         temp_data = pd.read_csv(os.path.join(root, file))
    #         print('source data:', len(temp_data))
    #         raw_data.append(temp_data)
    # raw_data = pd.concat(raw_data, sort=False)
    #
    # raw_data = raw_data[~raw_data['label_string'].isna()]
    # raw_data = raw_data[(~raw_data['proof'].isna())]
    # raw_data = raw_data.sample(frac=1).reset_index().drop('index', axis=1)[:30000]
    # raw_data = raw_data.drop('label_string', axis=1).join(
    #     raw_data['label_string'].str.split(';', expand=True).stack().reset_index(level=1, drop=True).rename('label_string'))
    # raw_data['suqiu'], raw_data['label'] = zip(*raw_data['label_string'].str.split(':'))
    # raw_data['new_problem'] = raw_data.apply(lambda row: suqiu_correct(row['suqing'], row['problem'], row['suqiu'])[0], axis=1)
    # raw_data['new_suqiu'] = raw_data.apply(lambda row: suqiu_correct(row['suqing'], row['problem'], row['suqiu'])[1], axis=1)

    raw_data = pd.read_csv('../data/prediction/result_train.csv', nrows=200000, usecols=['problem', 'suqiu', 'proof', 'serial', 'sucheng', 'chaming', 'renwei', 'panjue'])
    raw_data = raw_data[(~raw_data['proof'].isna())]
    print('raw data size %s' % (len(raw_data)))
    raw_data = raw_data.sample(frac=1).reset_index().drop('index', axis=1)

    # 特征匹配
    for problem, suqiu_list in logic_ps.items():
        if problem not in problem_list:
            continue
        data = raw_data[raw_data['problem']==problem].copy()
        print('data size:', len(data))

        data_x_kw = multi_processing_data(data['chaming'].fillna('').values,
                                          min(process_num, len(data)),
                                          problem, None)
        print(data_x_kw.shape)

        for i, f in enumerate(problem_bkw_dict[problem].index):
            data[f] = data_x_kw[:, i]

        data['matched_factor'] = data.apply(lambda row: '\n'.join(data.columns[row==1]), axis=1)

        for suqiu in suqiu_list:
            result = []
            for f in suqiu_bkw_dict[problem+'_'+suqiu].index:
                if len(data[data[f] == 1]) == 0:
                    print('not match', problem, f)
                    continue
                temp = data[(data['suqiu']==suqiu) & (data[f] == 1)].sample(frac=1)[:num_per_factor].copy()
                for index, row in temp.iterrows():
                    sentences = sentence_process(row['proof'].split('|'), max_seq_length)
                    if len(sentences) == 0:
                        print(row['proof'])
                    preds = predict(sentences, 32)
                    result.append([suqiu, f, row['proof'], '、'.join(preds), row['matched_factor'], row['serial'], row['sucheng'], row['chaming'], row['renwei']])

            for index, row in data.sample(frac=1)[-50:].iterrows():
                sentences = sentence_process(row['proof'].split('|'), max_seq_length)
                if len(sentences) == 0:
                    print(row['proof'])
                preds = predict(sentences, 32)
                result.append([None, None, row['proof'], '、'.join(preds), row['matched_factor'], row['serial'], row['sucheng'], row['chaming'], row['renwei']])

            result = pd.DataFrame(result, columns=['suqiu', 'factor', 'sentence', 'proof', 'matched_factor', 'serial', 'sucheng', 'chaming', 'renwei'])
            result.to_csv('../data/'+problem+'_'+suqiu+'证据.csv',index=False, encoding='utf-8')


if __name__=='__main__':
    proof_extract(['知识产权'])