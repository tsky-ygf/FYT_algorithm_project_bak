#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 09:29
# @Author  : Adolf
# @Site    : 
# @File    : text_error_correction.py
# @Software: PyCharm
# from pycorrector.macbert.macbert_corrector import MacBertCorrector
#
# if __name__ == '__main__':
#     error_sentences = [
#         '真麻烦你了。希望你们好好的跳无',
#         '少先队员因该为老人让坐',
#         '机七学习是人工智能领遇最能体现智能的一个分知',
#         '一只小鱼船浮在平净的河面上',
#         '我的家乡是有明的渔米之乡',
#     ]
#
#     m = MacBertCorrector("shibing624/macbert4csc-base-chinese")
#     for line in error_sentences:
#         correct_sent, err = m.macbert_correct(line)
#         print("query:{} => {}, err:{}".format(line, correct_sent, err))


from paddlenlp import Taskflow

text_correction = Taskflow("text_correction")
res1 = text_correction(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
# res2 = text_correction('人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。')

print(res1)
