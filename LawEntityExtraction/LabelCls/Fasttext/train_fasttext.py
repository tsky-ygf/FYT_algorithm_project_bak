#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 13:30
# @Author  : Adolf
# @Site    : 
# @File    : train_fasttext.py
# @Software: PyCharm
import sys

import jieba
import fasttext
import pandas as pd
from math import ceil
from sklearn.model_selection import train_test_split

from RelevantLaws.DataProcess.fasttext_train_data_process import read_stopwords
from fasttext_train_data_process import generate_train_data, generate_model_data

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


def train_fasttext_model(train_data_path, save_model_path):
    classifier = fasttext.train_supervised(train_data_path,
                                           label='__label__',
                                           thread=3,
                                           wordNgrams=3,
                                           min_count=5,
                                           epoch=25,
                                           lr=0.3,  # 0.3
                                           loss='softmax',#softmax hs
                                           dim=150,
                                           bucket=500000)

    # classifier.save_model('model/bxh_law_name_and_items/fasttext.bin')
    classifier.save_model(save_model_path)


def evaluate_result(train_data_path, dev_data_path, test_data_path, _model_path):
    # classifier = fasttext.load_model('model/bxh_law_name_and_items/fasttext.bin')
    classifier = fasttext.load_model(_model_path)

    train_result = classifier.test(train_data_path)
    # print(train_result)
    print('Number of train examples:', train_result[0])
    print('train_precision:', train_result[1])
    print('train_recall:', train_result[2])
    train_f1 = 2 * train_result[1] * train_result[2] / (train_result[1] + train_result[2])
    print('train_f1:', train_f1)

    dev_result = classifier.test(dev_data_path)
    # print(dev_result)
    print('Number of dev examples:', dev_result[0])
    print('dev_precision:', dev_result[1])
    print('dev_recall:', dev_result[2])
    dev_f1 = 2 * train_result[1] * train_result[2] / (dev_result[1] + dev_result[2])
    print('dev_f1:', dev_f1)
    #
    # test_result = classifier.test(test_data_path, k=3)
    # print('Number of test examples:', test_result[0])
    # print('test_precision:', test_result[1])
    # print('test_recall:', test_result[2])
    # test_f1 = 2 * (train_result[1] * train_result[2] )/ (test_result[1] + test_result[2])
    # print('test_f1:', test_f1)


def infer_result(_texts, _stopword):
    # 结果
    classifier = fasttext.load_model('model/bxh_law_name_and_items/loan_fasttext.bin')

    _texts = jieba.lcut(_texts)
    _texts = list(filter(lambda x: len(x) > 1, _texts))
    _texts = list(filter(lambda x: x not in _stopword, _texts))

    # texts = ''
    labels = classifier.predict(' '.join(_texts), k=5)
    print(labels)
    # print(labels[0][0].strip('__label__'))


if __name__ == '__main__':
    # train_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/train_data.txt'
    # dev_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/dev_data.txt'
    # test_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/test_data.txt'
    #
    # model_path = "model/bxh_law_name_and_items/civil_fasttext.bin"
    # train_fasttext_model(train_path, save_model_path=model_path)
    # evaluate_result(train_path, dev_path, test_path, _model_path=model_path)

    # data_path = 'LawEntityExtraction/data/bert/reduce_child_support/reduce_child_support.json'
    # # Tf-idf 拆分句子
    # sentences = generate_train_data(data_path)
    # label_sit_spe = []
    # for item_sentences in sentences:
    #     label_situa, word = item_sentences.split('\t')
    #     if label_situa not in label_sit_spe:
    #         label_sit_spe.append(label_situa)
    # data_situa_list = [[] for j in range(len(label_sit_spe))]
    # for item_sentences in sentences:
    #     label_situa, word = item_sentences.split('\t')
    #     if label_situa in label_sit_spe:
    #         data_situa_list[label_sit_spe.index(label_situa)].append(item_sentences)
    # augment_list = []
    # copy_list = []
    # split_list = []
    # for item_data_sit in data_situa_list:
    #     if len(item_data_sit) < 20:
    #         augment_list.append(item_data_sit)
    #     elif 20 < len(item_data_sit) < 205:
    #         copy_list.append(item_data_sit)
    #     else:
    #         split_list.append(item_data_sit)
    # min_len = sys.maxsize
    # for item_split in split_list:
    #     if len(item_split) < min_len:
    #         min_len = len(item_split)
    # split_num = 0
    # if min_len % 100 > 33:
    #     split_num = ceil(min_len / 100)
    # else:
    #     split_num = int(min_len / 100)
    # if min_len == sys.maxsize:
    #     split_num = 1
    # for i in range(split_num):
    #     final_list_temp = []
    #     data_train_path = 'LawEntityExtraction/data/fasttext/reduce_child_support/spec_train_ner' + str(i) + '.txt'
    #     data_dev_path = 'LawEntityExtraction/data/fasttext/reduce_child_support/spec_dev_ner' + str(i) + '.txt'
    #     for item_copy_list in copy_list:
    #         for sub_copy_list in item_copy_list:
    #             final_list_temp.append(sub_copy_list)
    #     for item_split_list in split_list:
    #         for sub_split_list in item_split_list[i * 100:(i + 1) * 100]:
    #             final_list_temp.append(sub_split_list)
    #     finalList = []
    #     label_final_s, words_final_s = [], []
    #     for item_final_list in final_list_temp:
    #         label_final, words_final = item_final_list.split('\t')
    #         label_final_s.append(label_final)
    #         words_final_s.append(words_final)
    #     df = pd.DataFrame({"label": label_final_s, "words": words_final_s})
    #     spec_train, spec_test = train_test_split(df, test_size=0.2, stratify=df["label"].values.tolist())
    #     spec_train_save = []
    #     spec_test_save = []
    #     for item_spec_train in spec_train.values.tolist():
    #         spec_train_save.append(item_spec_train[0] + '\t' + item_spec_train[1])
    #     for item_spec_test in spec_test.values.tolist():
    #         spec_test_save.append(item_spec_test[0] + '\t' + item_spec_test[1])
    #     generate_model_data(spec_train_save, data_train_path)
    #     generate_model_data(spec_test_save, data_dev_path)

    train_path = 'LawEntityExtraction/data/fasttext/reduce_child_support/spec_train_ner' + str(0) + '.txt'
    dev_path = 'LawEntityExtraction/data/fasttext/reduce_child_support/spec_dev_ner' + str(0) + '.txt'
    test_path = 'LawEntityExtraction/data/fasttext/reduce_child_support/spec_dev_ner' + str(0) + '.txt'

    model_path = "model/bxh_law_name_and_items/loan_fasttext.bin"
    train_fasttext_model(train_path, model_path)
    # evaluate_result(train_path, dev_path, _model_path=model_path)

    # model = fasttext.train_supervised(input=train_path,
    #                                   autotuneValidationFile=dev_path,
    #                                   autotuneDuration=300)
    # model.save_model(model_path)
    evaluate_result(train_path, dev_path, test_path, model_path)

    stopwords = read_stopwords("data/stopwords.txt")
    # text = '原告廖淦流诉称：原、被告是多年朋友。2013年3月7日，被告向原告借款150000元，约定2个月限期还款' \
    #        '，到期后经催收仍不归还。据此，请求法院判决被告清还原告本金150000元及利息33750元' \
    #        '（利息从2013年3月7日至2013年12月7日，按每月25％计算），案件诉讼费由被告承担。' \
    #        '被告彭志辉没有答辩也没有向本院提交相关证据。被告卢木兰没有答辩也没有向本院提交相关证据。' \
    #        '被告彭金星没有答辩也没有向本院提交相关证据。经审理查明：2013年3月7日，' \
    #        '被告彭志辉、彭金星向与原告借款150000元，并立下借据，约定月息5分。' \
    #        '后经原告催收，两被告没有还款。诉讼中，原告述称：与被告是朋友关系，借款是现金交付，' \
    #        '两被告到自己店铺取款，没有抵押物，最后一次向被告催收借款是2013年7月。' \
    #        '由于三被告均没有答辩及到庭参加诉讼，本案未能调解。又查明：被告彭志辉与被告卢木兰是夫妻。' \
    #        '2013年12月19日，原告向本院提出财产保全申请，要求查封被告卢木兰座落于四会市东城街道玉器城二座55号（五楼）房屋。' \
    #        '同年12月20日，本院作出（2013）肇四法民一初字第1038号之一民事裁定，' \
    #        '冻结被告彭志辉、卢木兰银行账户内存款183750元或查封、扣押相同价值财产，' \
    #        '查封原告享有的用作财产保全担保的位于四会市贞山街道独岗村委会下沙开发小区16号地的土地使用权。' \
    #        '以上事实，有原告提交的《起诉状》、本人《居民身份证》、《借据》、《结婚证》、《土地使用权证》，' \
    #        '本院制作的《询问笔录》、《开庭笔录》为证证实，予以认定。'
    # text = '原告高某1向本院提出诉讼请求：1、判令被告向原告支付抚养费每月1500元。事实及理由：' \
    #        '原告系被告亲生女儿，2015年11月17日，被告与原告之母因夫妻感情不和经昆明市西山区人民法院判决离婚，' \
    #        '原告由母亲抚养，被告每月支付抚养费800元，该抚养在当时仅够维持最低生活。' \
    #        '原告因要上学及生活开支的需要每月800元的抚养费早已不够，而原告生母的收入有限，' \
    #        '一直未再婚独自承担原告的抚养费用非常困难。被告虽有较高的经济收入，' \
    #        '但基于与生母的关系恶化，一直说无职业，拒不增加抚养费用，故提起诉讼。'
    text = '本院认为，离婚协议中关于财产分割的条款，对男女双方具有法律约束力。原、被告双方已于2020年9月3日办理离婚登记，并签订离婚协议书，就债务处理进行了约定。根据约定，针对横龙信用社贷款5万元，原、被告双方各承担2' \
           '.5万元，被告在离婚后一星期内将2.5' \
           '万元支付给原告。该协议系原、被告双方为达成离婚目的自愿签订的，原、被告均应按照该协议内容来履行，且该协议中约定的银行贷款是原、被告双方的名义共同所贷，现被告一直未按协议内容履行支付义务，故原告要求被告支付2.5' \
           '万元的诉讼主张，于法有据，本院予以支持。 '
    infer_result(text, stopwords)
    text_sub = text.split('。')
    for item in text_sub:
        infer_result(item, stopwords)
