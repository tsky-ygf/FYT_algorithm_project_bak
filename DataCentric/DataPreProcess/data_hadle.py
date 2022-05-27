#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 16:12
# @Author  : Adolf
# @Site    : 
# @File    : data_hadle.py
# @Software: PyCharm
from pathlib import Path
from DocumentReview.ParseFile.parse_word import read_docx_file

jt_path = Path('data/DocData/借条')

index = 0
for jt_file in list(jt_path.glob('*.docx')):
    # if jt_file.suffix == '.docx':
        # print(read_docx_file(jt_file))
        # break
    # print(jt_file)
    file = Path('data/DocData/result') / 'jietiao_{}.txt'.format(index)
    file.write_text("\n".join(read_docx_file(jt_file)))
    index += 1