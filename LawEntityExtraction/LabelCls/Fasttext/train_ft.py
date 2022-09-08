#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 13:30
# @Author  : Adolf
# @Site    :
# @File    : train_ft.py
# @Software: PyCharm

import jieba
import fasttext

from ProfessionalSearch.RelevantLaws.DataProcess.fasttext_train_data_process import read_stopwords

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
                                           loss='ova',
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
    train_f1 = 2 * (train_result[1] * train_result[2] )/ (train_result[1] + train_result[2])
    print('train_f1:', train_f1)

    dev_result = classifier.test(dev_data_path)
    # print(dev_result)
    print('Number of dev examples:', dev_result[0])
    print('dev_precision:', dev_result[1])
    print('dev_recall:', dev_result[2])
    dev_f1 = 2 * (train_result[1] * train_result[2] )/ (dev_result[1] + dev_result[2])
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

    train_path = 'LawEntityExtraction/data/fasttext/split_num_path/spec_train_ner' + str(0) + '.txt'
    dev_path = 'LawEntityExtraction/data/fasttext/split_num_path/spec_train_ner' + str(1) + '.txt'
    test_path = 'LawEntityExtraction/data/fasttext/split_num_path/spec_train_ner' + str(2) + '.txt'

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
    text = '原告郭啟兴、李宗秀向本院提出诉讼请求如下：1.请求判令被告郭子刚向二原告支付拖欠赡养费5200元；三被告每人每月向二原告给付赡养费200元，每年春节另给付200元，三被告每人每年共给付赡养费2600元；2' \
           '.案件受理费由三被告承担。事实和理由：二原告系夫妻关系，被告郭子刚系二原告的长子，二原告另有次子郭子亮、三子郭子川。二原告曾与三被告达成约定，三被告每人每月向二原告给付赡养费200元，每年春节另给付200' \
           '元，每人每年共给付赡养费2600元，赡养费于每年春节前给付。现二原告年事已高，无经济收入来源丧失了劳动能力，二原告需要子女履行赡养义务，然而被告郭子刚从2018年春节前就拒绝支付赡养费，现在又到了2019' \
           '年，被告郭子刚应给付二原告2018年和2019年的赡养费共5200' \
           '元。由于被告郭子刚不尽子女应尽的赡养义务。为维护原告的合法权益，故原告成讼。被告郭子刚辩称，不同意原告的诉讼请求，因为自身经济能力有限，每年最多给付二原告赡养费1500' \
           '元。被告郭子亮无答辩。被告郭子川辩称，同意原告的诉讼请求。本院经审理认定事实如下：原告郭啟兴、李宗秀系夫妻关系，其在婚姻关系存续期间生育三子二女，长子郭子刚，次子郭子亮，' \
           '三子郭子川，长女郭子红，次女郭子梅。现五个子女均已成家独立生活。原告郭啟兴现年82周岁，原告李宗秀现年83周岁，二人均年事已高。另查，二原告均为农业家庭户，被告郭子刚2018' \
           '年度未给付二原告赡养费，被告郭子亮、郭子川已按诉前原、被告约定给付二原告赡养费至2018年度。以上事实有原告提交的户口薄复印件，原告提交的天津市静海区梁头镇肖民庄村村民委员会证明一份以及庭审笔录为据。 '
    infer_result(text, stopwords)
    text_sub = text.split('。')
    for item in text_sub:
        infer_result(item, stopwords)
