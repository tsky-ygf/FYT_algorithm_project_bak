#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 10:16
# @Author  : Adolf
# @Site    : 
# @File    : data_prepare.py
# @Software: PyCharm
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

origin_data_path = "data/fyt_train_use_data/QA/origin_data.csv"


def create_label_map():
    df = pd.read_csv(origin_data_path)
    label_list = list(set(df.sub_source.values.tolist()))
    print(label_list)
    print(len(label_list))
    with open("IntelligentConsultation/config/query_cls_label.txt", "w") as f:
        for index, label in enumerate(label_list):
            f.write(f"{label},{index}\n")


def create_train_data():
    df = pd.read_csv(origin_data_path)

    label_map = {}
    with open("IntelligentConsultation/config/query_cls_label.txt", "r") as f:
        for line in f.readlines():
            label_map[line.split(",")[0]] = line.split(",")[1].strip()

    # print(label_map)

    data_list = []
    for index, row in tqdm(df.iterrows()):
        data_list.append([row.question, label_map[row.sub_source]])
        # if index > 20:
        #     break
    # print(data_list)
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    # print(train_data)
    # print(test_data)
    with open("data/fyt_train_use_data/query_cls/train_data.csv", "w") as f:
        for data in train_data:
            f.write(f"{data[0]}\t{data[1]}\n")

    with open("data/fyt_train_use_data/query_cls/test_data.csv", "w") as f:
        for data in test_data:
            f.write(f"{data[0]}\t{data[1]}\n")


if __name__ == '__main__':
    # create_label_map()
    create_train_data()
