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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import numpy as np

# from torch.multiprocessing import set_start_method
# set_start_method('spawn')

# num_cpus = psutil.cpu_count(logical=False)
# print(f"num_cpus: {num_cpus}")
# ray.init(num_cpus=16)
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
    # segs = get_fileter_data(_content_lines, _ner=ner, is_tokenization=True)
    segs = jieba.cut(_content_lines, use_paddle=True)
    segs = ' '.join(list(segs))
    segs = segs.split(' ')
    segs = list(filter(lambda x: len(x) > 1, segs))
    segs = list(filter(lambda x: x not in _stopwords, segs))
    category_str = ""
    for category in category_list:
        category_str += "__label__" + str(category) + "\t"

    return category_str + " ".join(segs)


# @ray.remote
def distributed_preprocess_text_text(one_judgment):
    facts = one_judgment['words']
    use_laws = one_judgment['labels']
    law_list = []
    for one_law in use_laws:
        if "诉讼" in one_law[0]['jiufen_type']:
            pass
        else:
            # law_list.append(one_law[0]['situation'] + '###' + one_law[1])
            law_list.append(one_law[1])

    # 拆分句子
    text = one_judgment['words']
    text_merge = ''
    text_sub = text.split('。')
    mention = one_judgment['mention']
    mention_sub = mention.split('。')
    if len(text_sub) == 1:
        text_merge = text
    elif len(mention_sub) == 1:
        for text_sub_item in text_sub:
            if mention in text_sub_item:
                text_merge = text_sub_item
    else:
        for mention_sub_item in mention_sub:
            for text_sub_item in text_sub:
                if len(mention_sub_item) > 1 and mention_sub_item in text_sub_item:
                    text_merge = text_merge + text_sub_item

    if not text_merge:
        print(text)
        return

    facts = text_merge
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


# @print_run_time
def generate_train_data(json_path):
    with open(json_path, 'rb') as f:
        # load_list = json.load(f)

    # pbar = tqdm(total=len(load_list))
        sentences_sep = []
        sentences = []
        for one_judgment in f:
            item_i = json.loads(one_judgment.strip())
            seg = distributed_preprocess_text_text(item_i)
            if seg is None:
                continue
            sentences.append(seg)
            sep = seg.split('\t')
            sentences_sep.append( [ sep[:len(sep)-1] ,  sep[len(sep)-1] ])

    # with pathos.multiprocessing.ProcessingPool(20) as p:
    #     sentences = list(tqdm(p.imap(distributed_preprocess_text_text, load_list), total=len(load_list), desc="获取数据"))

    # sentences = ray.get(
    # [distributed_preprocess_text_text.remote(index, one_judgment) for index, one_judgment in enumerate(load_list)])

    # sentences = [distributed_preprocess_text_text.remote(one_judgment) for one_judgment in load_list]
    # for x in tqdm(to_iterator(sentences), total=len(sentences)):
    #     pass
    label_list = [i[0] for i in sentences_sep]
    vectorizer = CountVectorizer()
    word_frequence = vectorizer.fit_transform([i[len(i) - 1] for i in sentences_sep])
    words = vectorizer.get_feature_names_out()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(word_frequence)
    weight = tfidf.toarray()

    n = 200  # 前200
    new_sentences = []
    for label_l, w in zip(label_list,weight):
        # 排序
        loc = np.argsort(-w)
        new_words = []
        for i in range(n):
            if w[loc[i]] == 0.0: # 前200 中 去掉为零的权重，如过没有，就200项
                break
            new_words.append(words[loc[i]])
        #     print(str(i + 1) +"---"+ '\t'.join(label_l)+"--" + str(words[loc[i]]) + "--" + str(w[loc[i]]))
        # print('----')
        sub_sentence = '\t'.join(label_l) + '\t' + ' '.join(new_words)
        # print(sub_sentence)
        new_sentences.append(sub_sentence)
    random.shuffle(new_sentences)
    return new_sentences
    # generate_model_data(new_sentences, save_path)


if __name__ == '__main__':
    # train_json_path = "data/fyt_train_use_data/CAIL-Long/civil/train.json"
    # dev_json_path = "data/fyt_train_use_data/CAIL-Long/civil/dev.json"
    # test_json_path = "data/fyt_train_use_data/CAIL-Long/civil/test.json"

    # train_json_path = "data/fyt_train_use_data/CAIL-Long/criminal/train.json"
    # dev_json_path = "data/fyt_train_use_data/CAIL-Long/criminal/dev.json"
    # test_json_path = "data/fyt_train_use_data/CAIL-Long/criminal/test.json"

    # train_save_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/train_data_ner.txt'
    # dev_save_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/dev_data_ner.txt'
    # test_save_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/test_data_ner.txt'

    # train_save_path = 'data/fyt_train_use_data/law_items_data/criminal_fasttext_use_small/train_data.txt'
    # dev_save_path = 'data/fyt_train_use_data/law_items_data/criminal_fasttext_use_small/dev_data.txt'
    # test_save_path = 'data/fyt_train_use_data/law_items_data/criminal_fasttext_use_small/test_data.txt'

    train_json_path = "LawEntityExtraction/data/fasttext/spec_situa_pre/train.json"
    dev_json_path = "LawEntityExtraction/data/fasttext/spec_situa_pre/dev.json"
    # test_json_path = "data/fyt_train_use_data/CAIL-Long/civil/test.json"

    train_save_path = 'LawEntityExtraction/data/fasttext/spec_situa_post/train_data_ner.txt'
    dev_save_path = 'LawEntityExtraction/data/fasttext/spec_situa_post/dev_data_ner.txt'
    # test_save_path = 'data/fyt_train_use_data/law_items_data/civil_fasttext_use_small/test_data_ner.txt'

    generate_train_data(train_json_path, train_save_path)
    generate_train_data(dev_json_path, dev_save_path)
    # generate_train_data(test_json_path, test_save_path)
