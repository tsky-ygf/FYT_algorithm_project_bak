#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 16:12
# @Author  : Adolf
# @Site    : 
# @File    : data_hadle.py
# @Software: PyCharm
from pathlib import Path

import pandas as pd

from DocumentReview.ParseFile.parse_word import read_docx_file


def trans_docx_to_txt(docx_path):
    docx_path = Path(docx_path)
    index = 0
    for jt_file in list(docx_path.glob('*.docx')):
        # if jt_file.suffix == '.docx':
        # print(read_docx_file(jt_file))
        # break
        # print(jt_file)
        file = Path('data/DocData/result2') / 'jietiao_{}.txt'.format(index)
        file.write_text("\n".join(read_docx_file(jt_file)))
        index += 1


def trans_csv_txt(csv_path):
    df = pd.read_csv(csv_path)
    df = df[["event_person", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "court_deside"]]
    df.fillna("", inplace=True)
    for index, row in df.iterrows():
        row_list = [r for r in row.tolist() if r != ""]
        row_str = "\n".join(row_list)
        print(row_str)
        file = Path('data/DocData/result3') / 'xz_{}.txt'.format(index)
        file.write_text(row_str)
        if index > 100:
            break
        # break
    # print(df)


trans_csv_txt("data/DocData/origin.csv")
