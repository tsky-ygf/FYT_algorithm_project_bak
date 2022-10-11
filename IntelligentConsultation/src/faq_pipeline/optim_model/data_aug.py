#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 13:26
# @Author  : Adolf
# @Site    : 
# @File    : data_aug.py
# @Software: PyCharm
import random


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


if __name__ == '__main__':
    WR_res = word_repetition(text="公司交不起税了怎么办", dup_rate=0.3)
    print(WR_res)
