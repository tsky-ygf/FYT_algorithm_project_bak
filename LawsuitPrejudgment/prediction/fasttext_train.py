# coding=utf-8
import fasttext
import pandas as pd
import jieba
import os
import sys
from LawsuitPrejudgment.common import prob_ps_desc, chaming_fact_extract


problem_list = ['婚姻继承', '劳动社保', '借贷纠纷', '交通事故']


def data_prepare(set_type):
    data = pd.read_csv('../data/prediction/result_'+set_type+'.csv', usecols=['problem','suqiu','chaming','label'])
    data = data[data['problem'].isin(problem_list)]
    data['chaming_fact'] = data['chaming'].apply(chaming_fact_extract)
    data = data[data['chaming_fact'].str.len()>5]
    data['ps'] = data['problem'] + '_' + data['suqiu']
    data['suqiu_desc'] = data['ps'].apply(lambda x: prob_ps_desc[x])
    data['content'] = data['suqiu_desc']+'。'+data['chaming_fact']
    data['words'] = data['content'].apply(lambda x: ' '.join(jieba.lcut(x)))

    file = open('../data/prediction/fasttext_'+set_type+'.txt', 'w', encoding='utf-8')
    for index, row in data.iterrows():
        file.write(row['words'] + ' __label__' + str(row['label']) + '\n')
    file.close()


def train_eval():
    model = fasttext.supervised('../data/prediction/fasttext_train.txt', '../model/ft/ft',
                                label_prefix="__label__", epoch=10,
                                word_ngrams=3, min_count=5, bucket=500000,
                                lr=0.1, silent=0, loss='softmax')

    train_result = model.test('../data/prediction/fasttext_train.txt')
    print("FastText acc(train):", train_result.precision, train_result.recall)
    valid_result = model.test('../data/prediction/fasttext_valid.txt')
    print("FastText acc(valid):", valid_result.precision, valid_result.recall)


if __name__=='__main__':
    data_prepare('train')
    data_prepare('valid')
    train_eval()
