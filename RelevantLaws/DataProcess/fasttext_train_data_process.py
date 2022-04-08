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
# import ray
# import psutil
import pathos.multiprocessing
from Utils.logger import print_run_time
from RelevantLaws.DataProcess.data_process import get_fileter_data
import hanlp

# from torch.multiprocessing import set_start_method
# set_start_method('spawn')

# num_cpus = psutil.cpu_count(logical=False)
# print(f"num_cpus: {num_cpus}")
# ray.init(num_cpus=16)

stopwords_path = "data/stopwords.txt"


# def to_iterator(obj_ids):
#     while obj_ids:
#         done, obj_ids = ray.wait(obj_ids)
#         yield ray.get(done[0])


def read_stopwords(_stopwords_path):
    _stopwords = pd.read_csv(_stopwords_path, index_col=False, quoting=3, sep="\t", names=['stopword'],
                             encoding='utf-8')
    _stopwords = _stopwords['stopword'].values
    # print(stopwords)
    return _stopwords


stopwords = read_stopwords(stopwords_path)

# 分词去停用词，并整理为fasttext要求的文本格式,没有加入数据处理的
# def preprocess_text(_content_lines, category_list, _stopwords: List):
#     segs = jieba.lcut(_content_lines)
#     segs = list(filter(lambda x: len(x) > 1, segs))
#     segs = list(filter(lambda x: x not in _stopwords, segs))
#     category_str = ""
#     for category in category_list:
#         # _sentences.append("__label__" + str(category) + "\t" + " ".join(segs))
#         category_str += "__label__" + str(category) + "\t"
#     return category_str + " ".join(segs)

ner = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)


# 使用ner进行部分去除后的分词
def preprocess_text(_content_lines, category_list, _stopwords: List):
    segs = get_fileter_data(_content_lines, _ner=ner, is_tokenization=True)
    segs = segs.split(' ')
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in _stopwords, segs))
    category_str = ""
    for category in category_list:
        category_str += "__label__" + str(category) + "\t"
    return category_str + " ".join(segs)


# @ray.remote
def distributed_preprocess_text_text(one_judgment):
    facts = one_judgment['fact']
    use_laws = one_judgment['laws']
    law_list = []
    for one_law in use_laws:
        if "诉讼" in one_law[0]['title']:
            pass
        else:
            law_list.append(one_law[0]['title'] + '###' + one_law[1])

    try:
        seg = preprocess_text(facts, law_list, stopwords)
    except Exception as e:
        print(e)
        # print(one_judgment)
        seg = ""
    return seg


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


@print_run_time
def generate_train_data(json_path, save_path):
    with open(json_path, 'rb') as f:
        load_list = json.load(f)

    # pbar = tqdm(total=len(load_list))

    sentences = []
    for one_judgment in tqdm(load_list):
        seg = distributed_preprocess_text_text(one_judgment)
        sentences.append(seg)

    # with pathos.multiprocessing.ProcessingPool(20) as p:
    #     sentences = list(tqdm(p.imap(distributed_preprocess_text_text, load_list), total=len(load_list), desc="获取数据"))

    # sentences = ray.get(
    # [distributed_preprocess_text_text.remote(index, one_judgment) for index, one_judgment in enumerate(load_list)])

    # sentences = [distributed_preprocess_text_text.remote(one_judgment) for one_judgment in load_list]
    # for x in tqdm(to_iterator(sentences), total=len(sentences)):
    #     pass

    random.shuffle(sentences)
    generate_model_data(sentences, save_path)


if __name__ == '__main__':
    train_json_path = "data/fyt_train_use_data/CAIL-Long/civil/train.json"
    dev_json_path = "data/fyt_train_use_data/CAIL-Long/civil/dev.json"
    test_json_path = "data/fyt_train_use_data/CAIL-Long/civil/test.json"

    # train_json_path = "data/fyt_train_use_data/CAIL-Long/criminal/train.json"
    # dev_json_path = "data/fyt_train_use_data/CAIL-Long/criminal/dev.json"
    # test_json_path = "data/fyt_train_use_data/CAIL-Long/criminal/test.json"

    train_save_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/train_data_ner.txt'
    dev_save_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/dev_data_ner.txt'
    test_save_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/test_data_ner.txt'

    # train_save_path = 'data/fyt_train_use_data/law_items_data/criminal_fasttext_use_small/train_data.txt'
    # dev_save_path = 'data/fyt_train_use_data/law_items_data/criminal_fasttext_use_small/dev_data.txt'
    # test_save_path = 'data/fyt_train_use_data/law_items_data/criminal_fasttext_use_small/test_data.txt'

    generate_train_data(train_json_path, train_save_path)
    generate_train_data(dev_json_path, dev_save_path)
    generate_train_data(test_json_path, test_save_path)
