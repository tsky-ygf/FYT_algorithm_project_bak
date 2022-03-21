#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 16:11
# @Author  : Adolf
# @Site    : 
# @File    : fasttext_train.py
# @Software: PyCharm
import os

import jieba
import pandas as pd
import random
import fasttext
from pathlib import Path
from typing import List
from tqdm.auto import tqdm

# stopwords_path = "Data/stopwords.txt"
#
#
# def read_stopwords(_stopwords_path):
#     _stopwords = pd.read_csv(_stopwords_path, index_col=False, quoting=3, sep="\t", names=['stopword'],
#                              encoding='utf-8')
#     _stopwords = _stopwords['stopword'].values
#     # print(stopwords)
#     return _stopwords
#
#
# stopwords = read_stopwords(stopwords_path)
#
#
# # 分词去停用词，并整理为fasttext要求的文本格式
# def preprocess_text(_content_lines, _sentences, category, _stopwords=List):
#     segs = jieba.lcut(_content_lines)
#     segs = list(filter(lambda x: len(x) > 1, segs))
#     segs = list(filter(lambda x: x not in _stopwords, segs))
#     _sentences.append("__label__" + str(category) + "\t" + " ".join(segs))
#
#
# # 生成训练数据
# sentences = []
# p_file = Path('Data/THUCNews')
# for file_dir in p_file.iterdir():
#     print(file_dir)
#     # exit()
#     # progress_bar = tqdm(range(len(file_dir.rglob('*.txt'))),
#     #                     disable=file_dir.name)
#
#     for file in file_dir.iterdir():
#         text = file.read_text()
#         content_lines = text.replace('\n', '').replace(' ', '').replace('/t', '')
#         # print(file_dir.name)
#         preprocess_text(_content_lines=content_lines, _sentences=sentences, category=file_dir.name,
#                         _stopwords=stopwords)
#         # progress_bar.update(1)
#
# # 数据打乱
# random.shuffle(sentences)
#
#
# # 写入数据-fasttext格式
# def generate_model_data(sentences):
#     train_num = int(len(sentences) * 0.8)
#     train_set = sentences[0:train_num]
#     test_set = sentences[train_num:-1]
#     print("writing data to fasttext format...")
#     with open('Data/train_data.txt', 'w') as out:
#         for sentence in train_set:
#             out.write(sentence + "\n")
#         print("done!")
#     with open('Data/test_data.txt', 'w') as f:
#         for sentence in test_set:
#             f.write(sentence + '\n')
#         print('done!')
#
#
# generate_model_data(sentences)
#
# classifier = fasttext.train_supervised('Data/train_data.txt', label='__label__', wordNgrams=2, epoch=20, lr=0.1,
#                                        dim=100)

# 参数说明
"""
训练一个监督模型, 返回一个模型对象
@param input: 训练数据文件路径
@param lr:              学习率
@param dim:             向量维度
@param ws:              cbow模型时使用
@param epoch:           次数
@param minCount:        词频阈值, 小于该值在初始化时会过滤掉
@param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
@param minn:            构造subword时最小char个数
@param maxn:            构造subword时最大char个数
@param neg:             负采样
@param wordNgrams:      n-gram个数
@param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
@param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
@param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
@param lrUpdateRate:    学习率更新
@param t:               负采样阈值
@param label:           类别前缀
@param verbose:         ??
@param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
@return model object
"""

# classifier.save_model('Model/fasttext.bin')

# train_result = classifier.test('Data/train_data.txt')
# print('train_precision:', train_result[1])
# print('train_recall:', train_result[2])
# print('Number of train examples:', train_result[0])
# test_result = classifier.test('Data/test_data.txt')
# print('test_precision:', test_result[1])
# print('test_recall:', test_result[2])
# print('Number of test examples:', test_result[0])

# label_to_cate = {1: 'technology', 2: 'car', 3: 'entertainment', 4: 'military', 5: 'sports'}
fasttext.FastText.eprint = lambda x: None
classifier = fasttext.load_model('Model/fasttext.bin')
texts = '中新网 日电 2018 预赛 亚洲区 强赛 中国队 韩国队 较量 比赛 上半场 分钟 主场 作战 中国队 率先 打破 场上 僵局 利用 角球 机会 大宝 前点 攻门 得手 中国队 领先'
# texts = '这 是 中国 第 一 次 军舰 演习'
labels = classifier.predict(texts)
print(labels)
print(labels[0][0].strip('__label__'))
