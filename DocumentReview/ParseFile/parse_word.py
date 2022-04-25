#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 10:21
# @Author  : Adolf
# @Site    : 
# @File    : parse_word.py
# @Software: PyCharm
from docx import Document


# 读取docx 文件
def read_docx_file(docx_path):
    document = Document(docx_path)
    all_paragraphs = document.paragraphs

    return_text_list = []

    for index, paragraph in enumerate(all_paragraphs):
        one_text = paragraph.text.replace(" ", "").replace("\u3000", "")
        if one_text != "":
            return_text_list.append(one_text)
    # print(return_text_list)
    return return_text_list


if __name__ == '__main__':
    res = read_docx_file(docx_path="data/DocData/LaborContract/劳动合同.docx")
    print(res)