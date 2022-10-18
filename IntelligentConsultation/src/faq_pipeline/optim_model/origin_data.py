#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/17 10:01
# @Author  : Adolf
# @Site    : 
# @File    : origin_data.py
# @Software: PyCharm
import json
import pandas as pd

pd.set_option('display.max_columns', None)


def get_origin_data():
    law_data_path = "data/fyt_train_use_data/QA/hualv_qa.json"

    source_list = []
    sub_source_list = []
    question_list = []
    answer_list = []

    with open(law_data_path, "r") as file:
        for line in file.readlines():
            dic = json.loads(line)
            # print(dic)
            if dic["type"] is None:
                source_list.append("未知")
            else:
                source_list.append(dic["type"])

            if dic["sub_type"] is None:
                sub_source_list.append("未知")
            else:
                sub_source_list.append(dic["sub_type"])

            question_list.append(dic["query"])
            answer_list.append(dic["answer"])
            # break

    topic_data_path = "data/fyt_train_use_data/QA/pro_qa.csv"
    topic_data = pd.read_csv(topic_data_path)

    # print(topic_data.head())
    for index, row in topic_data.iterrows():
        # print(row)
        source_list.append("专题")
        sub_source_list.append(row["type"])
        question_list.append(row["question"])
        answer_list.append(row["answer"])
        # break

    # print(query_list)
    source_list = [source.replace(" ", "") for source in source_list]
    sub_source_list = [sub_source.replace(" ", "") for sub_source in sub_source_list]
    question_list = [query.replace(" ", "") for query in question_list]
    answer_list = [answer.replace(" ", "") for answer in answer_list]

    final_data_df = pd.DataFrame(
        {"source": source_list, "sub_source": sub_source_list, "question": question_list, "answer": answer_list})

    # print(final_data_df)
    final_data_df.to_csv("data/fyt_train_use_data/QA/origin_data.csv", index=False)


def deduplication_data():
    origin_data = pd.read_csv("data/fyt_train_use_data/QA/origin_data.csv")
    origin_data.drop_duplicates(subset="question", keep='first', inplace=True)

    origin_data.to_csv("data/fyt_train_use_data/QA/origin_data.csv", index=False)


if __name__ == '__main__':
    deduplication_data()
