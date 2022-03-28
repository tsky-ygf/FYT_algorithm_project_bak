#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 10:06
# @Author  : Adolf
# @Site    : 
# @File    : fasttext_train_data_process.py
# @Software: PyCharm
import json
import pandas as pd
import jieba
from typing import List
import random
from tqdm.auto import tqdm

stopwords_path = "data/fyt_train_use_data/stopwords.txt"


def read_stopwords(_stopwords_path):
    _stopwords = pd.read_csv(_stopwords_path, index_col=False, quoting=3, sep="\t", names=['stopword'],
                             encoding='utf-8')
    _stopwords = _stopwords['stopword'].values
    # print(stopwords)
    return _stopwords


stopwords = read_stopwords(stopwords_path)


# 分词去停用词，并整理为fasttext要求的文本格式
def preprocess_text(_content_lines, category_list, _stopwords: List):
    segs = jieba.lcut(_content_lines)
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in _stopwords, segs))
    category_str = ""
    for category in category_list:
        # _sentences.append("__label__" + str(category) + "\t" + " ".join(segs))
        category_str += "__label__" + str(category) + "\t"
    return category_str + " ".join(segs)


# 写入数据-fasttext格式
def generate_model_data(sentences, save_path):
    # train_num = int(len(sentences) * 0.8)
    # train_set = sentences[0:train_num]
    # test_set = sentences[train_num:-1]
    print("writing data to fasttext format...")
    # with open('Data/train_data.txt', 'w') as out:
    with open(save_path, 'w') as out:
        for sentence in sentences:
            out.write(sentence + "\n")
        print("done!")


def generate_train_data(json_path, save_path):
    with open(json_path, 'rb') as f:
        load_list = json.load(f)

    sentences = []
    for one_judgment in tqdm(load_list):
        facts = one_judgment['fact']
        use_laws = one_judgment['laws']
        law_list = []
        for one_law in use_laws:
            if "诉讼" in one_law[0]['title']:
                pass
            else:
                law_list.append(one_law[0]['title'] + '###' + one_law[1])
        # break
        seg = preprocess_text(facts, law_list, stopwords)
        # print(seg)
        sentences.append(seg)
        # break
    random.shuffle(sentences)
    generate_model_data(sentences, save_path)


if __name__ == '__main__':
    train_json_path = "data/fyt_train_use_data/CAIL-Long/civil/train.json"
    dev_json_path = "data/fyt_train_use_data/CAIL-Long/civil/dev.json"
    test_json_path = "data/fyt_train_use_data/CAIL-Long/civil/test.json"

    train_save_path = 'data/fyt_train_use_data/law_items_data/fasttext_use_small/train_data.txt'
    dev_save_path = 'data/fyt_train_use_data/law_items_data/fasttext_use_small/dev_data.txt'
    test_save_path = 'data/fyt_train_use_data/law_items_data/fasttext_use_small/test_data.txt'

    generate_train_data(train_json_path, train_save_path)
    generate_train_data(dev_json_path, dev_save_path)
    generate_train_data(test_json_path, test_save_path)
