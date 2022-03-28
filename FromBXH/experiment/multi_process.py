#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 20:42
# @Author  : Adolf
# @Site    : 
# @File    : multi_process.py
# @Software: PyCharm
import multiprocessing
from functools import partial

BUCKET_SIZE = 50000


def run_process(df, start):
    df = df[start:start + BUCKET_SIZE]
    print(start, "to ", start + BUCKET_SIZE)
    temp = df["question"].apply(preprocess)


chunks = [x for x in range(0, df.shape[0], BUCKET_SIZE)]
pool = multiprocessing.Pool()
func = partial(run_process, df)
temp = pool.map(func, chunks)
pool.close()
pool.join()
