#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 13:30
# @Author  : Adolf
# @Site    : 
# @File    : train_fasttext.py
# @Software: PyCharm
import jieba
import fasttext
from RelevantLaws.DataProcess.fasttext_train_data_process import read_stopwords

fasttext.FastText.eprint = lambda x: None

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


def train_fasttext_model(train_data_path):
    classifier = fasttext.train_supervised(train_data_path,
                                           label='__label__',
                                           wordNgrams=3,
                                           min_count=5,
                                           epoch=20,
                                           lr=0.3,
                                           bucket=500000)

    classifier.save_model('model/bxh_law_name_and_items/fasttext.bin')


def evaluate_result(train_data_path, dev_data_path, test_data_path):
    classifier = fasttext.load_model('model/bxh_law_name_and_items/fasttext.bin')

    train_result = classifier.test(train_data_path, k=3)
    print('train_precision:', train_result[1])
    print('train_recall:', train_result[2])
    print('Number of train examples:', train_result[0])

    dev_result = classifier.test(dev_data_path, k=3)
    print('dev_precision:', dev_result[1])
    print('dev_recall:', dev_result[2])
    print('Number of train examples:', dev_result[0])

    test_result = classifier.test(test_data_path, k=3)
    print('test_precision:', test_result[1])
    print('test_recall:', test_result[2])
    print('Number of test examples:', test_result[0])


def infer_result(_texts, _stopword):
    # 结果
    classifier = fasttext.load_model('model/bxh_law_name_and_items/fasttext.bin')

    _texts = jieba.lcut(_texts)
    _texts = list(filter(lambda x: len(x) > 1, _texts))
    _texts = list(filter(lambda x: x not in _stopword, _texts))

    # texts = ''
    labels = classifier.predict(' '.join(_texts),k=3)
    print(labels)
    # print(labels[0][0].strip('__label__'))


if __name__ == '__main__':
    train_path = 'data/fyt_train_use_data/law_items_data/fasttext_use_small/train_data.txt'
    dev_path = 'data/fyt_train_use_data/law_items_data/fasttext_use_small/dev_data.txt'
    test_path = 'data/fyt_train_use_data/law_items_data/fasttext_use_small/test_data.txt'

    # train_fasttext_model(train_path)
    # evaluate_result(train_path, dev_path, test_path)

    stopwords = read_stopwords("data/fyt_train_use_data/stopwords.txt")
    text = '原告廖淦流诉称：原、被告是多年朋友。2013年3月7日，被告向原告借款150000元，约定2个月限期还款' \
           '，到期后经催收仍不归还。据此，请求法院判决被告清还原告本金150000元及利息33750元' \
           '（利息从2013年3月7日至2013年12月7日，按每月25％计算），案件诉讼费由被告承担。' \
           '被告彭志辉没有答辩也没有向本院提交相关证据。被告卢木兰没有答辩也没有向本院提交相关证据。' \
           '被告彭金星没有答辩也没有向本院提交相关证据。经审理查明：2013年3月7日，' \
           '被告彭志辉、彭金星向与原告借款150000元，并立下借据，约定月息5分。' \
           '后经原告催收，两被告没有还款。诉讼中，原告述称：与被告是朋友关系，借款是现金交付，' \
           '两被告到自己店铺取款，没有抵押物，最后一次向被告催收借款是2013年7月。' \
           '由于三被告均没有答辩及到庭参加诉讼，本案未能调解。又查明：被告彭志辉与被告卢木兰是夫妻。' \
           '2013年12月19日，原告向本院提出财产保全申请，要求查封被告卢木兰座落于四会市东城街道玉器城二座55号（五楼）房屋。' \
           '同年12月20日，本院作出（2013）肇四法民一初字第1038号之一民事裁定，' \
           '冻结被告彭志辉、卢木兰银行账户内存款183750元或查封、扣押相同价值财产，' \
           '查封原告享有的用作财产保全担保的位于四会市贞山街道独岗村委会下沙开发小区16号地的土地使用权。' \
           '以上事实，有原告提交的《起诉状》、本人《居民身份证》、《借据》、《结婚证》、《土地使用权证》，' \
           '本院制作的《询问笔录》、《开庭笔录》为证证实，予以认定。'

    infer_result(text, stopwords)
