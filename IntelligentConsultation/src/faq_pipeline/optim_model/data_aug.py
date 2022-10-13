#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 13:26
# @Author  : Adolf
# @Site    : 
# @File    : data_aug.py
# @Software: PyCharm
import random
import requests

import pandas as pd
from jieba import analyse


def word_repetition(text, dup_rate=0.32):
    """
    词重复
    :param text:
    :param dup_rate:
    :return:
    """
    text_list = []
    for word in text:
        text_list.append(word)
        if random.random() < dup_rate:
            text_list.append(word)

    wr_word = ''.join(text_list)
    return wr_word


def random_reverse_order(text):
    a, b = random.sample(text, 2)
    text_list = list(text)
    a_index = text_list.index(a)
    b_index = text_list.index(b)

    text_list[a_index], text_list[b_index] = text_list[b_index], text_list[a_index]
    text = ''.join(text_list)
    return text


def get_all_keywords():
    tfidf = analyse.extract_tags

    train_data_path = "data/fyt_train_use_data/QA/pro_qa.csv"
    data = pd.read_csv(train_data_path)
    questions = data.question.tolist()

    for query in questions:
        keywords = tfidf(query, allowPOS=["n", "v", "a", "d"])
        print(f"query:{query} ======> keywords:{keywords}")


def get_keywords_and_synonym():
    # tfidf = analyse.extract_tags
    # keywords = tfidf(text)
    # print(keywords)
    # for word in keywords:
    #     res = requests.get(f"https://kmcha.com/api/similar/{word}").json()
    #     print(res)

    train_data_path = "data/fyt_train_use_data/QA/pro_qa.csv"
    data = pd.read_csv(train_data_path)
    questions = data.question.tolist()
    # print(questions)

    all_keyword = []

    tfidf = analyse.extract_tags

    for query in questions:
        keywords = tfidf(query)
        for keyword in keywords:
            if keyword not in all_keyword:
                all_keyword.append(keyword)

    # print(all_keyword)
    # print(len(all_keyword))
    word_list = []
    similar_list = []
    for word in all_keyword:
        res = requests.get(f"https://kmcha.com/api/similar/{word}").json()
        # print(res)
        if len(res["similar"]) > 0:
            word_list.append(word)
            similar_list.append("|".join(res["similar"]))
        # break
    res_df = {"word": word_list, "similar": similar_list}
    res_df = pd.DataFrame(res_df)
    res_df.to_csv("data/fyt_train_use_data/QA/word_similar.csv", index=False)


def filter_similar_word():
    similar_df = pd.read_csv("data/fyt_train_use_data/QA/word_similar.csv")

    for index, row in similar_df.iterrows():
        # print(index, row)
        similar_list = row["similar"].split("|")
        new_similar_list = []
        word = row["word"]
        for similar_word in similar_list:
            if word not in similar_word:
                new_similar_list.append(similar_word)
        similar_df.loc[index, "similar"] = "|".join(new_similar_list)

    similar_df.to_csv("data/fyt_train_use_data/QA/word_similar.csv", index=False)


if __name__ == '__main__':
    _text = "公司交不起税了怎么办"
    # WR_res = word_repetition(text="公司交不起税了怎么办", dup_rate=0.3)
    # print(WR_res)
    # get_keywords_and_synonym()
    # rest = random_reverse_order(text=_text)
    # print(rest)
    # filter_similar_word()
    get_all_keywords()
