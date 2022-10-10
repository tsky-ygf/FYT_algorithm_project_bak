# -*- coding: utf-8 -*-
import pandas as pd
import re
import os

problem_list = ['婚姻家庭']
max_seq_length = 64


def sentence_labelling(sentences, pattern, drop_unmatch):
    """
    基于打标的数据对句子进行标注
    :param sentences:
    :param pattern:
    :param drop_unmatch:
    :return:
    """
    data = []
    for sentence in sentences:
        labels = re.findall(pattern, sentence)
        if drop_unmatch and len(labels) == 0:
            continue
        temp = ['O'] * len(sentence)
        for label in labels:
            start_index = sentence.index(label)
            if temp[start_index] != 'O':
                continue
            if len(label) == 1:
                temp[start_index] = 'S'
            else:
                temp[start_index] = 'B'
                for i in range(1, len(label) - 1):
                    temp[start_index + i] = 'M'
                temp[start_index + len(label) - 1] = 'E'
                i = 0
                while start_index + len(label) + i<len(temp) and temp[start_index + len(label) + i] in ['M', 'E']:
                    temp[start_index + len(label) + i] = 'O'
                    i += 1
        data.append([sentence, '、'.join(labels), ''.join(temp).replace('EB', 'MM')])
    return pd.DataFrame(data, columns=['sentence', 'patterns', 'label'])


def sentence_process(sentences, max_seq_length):
    """
    基于最大长度对数据进行截断
    :param sentences:
    :param max_seq_length:
    :return:
    """
    result = []
    for sentence in sentences:
        if len(sentence) < max_seq_length:
            result.append(sentence)
        else:
            temp = ''
            for sen in re.split('[。；;]', sentence):
                if len(temp + sen) > max_seq_length:
                    result.append(temp)
                    temp = sen
                else:
                    temp += sen + '。'
            result.append(temp)
    return result


def get_extra_data(problem, extra_num):
    """
    读取其他数据集
    :param problem:
    :param extra_num:
    :return:
    """
    extra_data = []
    for root, dirs, files in os.walk('/datadisk2/tyin/20190328/'):
        for file in files:
            print(os.path.join(root, file))
            if problem not in file:
                continue
            temp_data = pd.read_csv(os.path.join(root, file))
            print('source data:', len(temp_data))
            extra_data.append(temp_data)
    extra_data = pd.concat(extra_data, sort=False)
    extra_data = extra_data[~extra_data['label_string'].isna()]
    extra_data = extra_data[~extra_data['proof'].isna()][:extra_num]
    print('data size:', len(extra_data))
    return extra_data


def ner_data_prepare(max_seq_length, problem, extra_num):
    """
    证据提取NER模型训练数据准备
    :param max_seq_length:
    :param problem:
    :param extra_num:
    """
    # 证据打标正则
    with open('../data/proof/' + problem + '证据打标.txt', 'r', encoding='utf-8') as file:
        pattern = [line.strip() for line in file.readlines()]
    pattern = '(' + '|'.join(pattern).replace('.*', '[^,;:，：。；？！（）、《》\(\)\s]') + ')'

    # 打标数据
    sentences = []
    with open('../data/proof/' + problem + '证据数据.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            sentences += line.strip().split('|')
    sentences = sentence_process(sentences, max_seq_length)
    data1 = sentence_labelling(sentences, pattern, False)

    # 增加其他数据
    extra_data = get_extra_data(problem, extra_num)
    sentences = []
    for line in extra_data['proof'].values:
        sentences += line.strip().split('|')
    sentences = sentence_process(sentences, max_seq_length)
    data2 = sentence_labelling(sentences, pattern, True)

    # 数据切分
    data = pd.concat([data1, data2])
    data = data[data['sentence'].str.len()>0]
    data = data.sample(frac=1)
    train_size = int(len(data) * 0.9)
    data[:train_size].to_csv('../data/proof/train.csv', index=False, encoding='utf-8')
    data[train_size:].to_csv('../data/proof/valid.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    # sentences = '上述事实，有原告陈述、被告书面答辩意见、（2014）郯民初字第1699号民事判决书、房权证等证据'
    # print(sentence_process(sentences.split('|'), 48))
    ner_data_prepare(max_seq_length, '婚姻家庭', 50000)
