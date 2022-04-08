#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/2 09:32
# @Author  : Adolf
# @Site    : 
# @File    : mapping_the_original_statute.py
# @Software: PyCharm
import pandas as pd

civil_law_path = 'data/law/整理过法条/民法典法条终版.csv'

df = pd.read_csv(civil_law_path)
df = df[["problem", "suqiu", "part", "chapter", "clause", "content", "备注"]]
df = df[df["备注"] == 1]
# print(df.head())
print(len(df))
