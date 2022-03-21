#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:04
# @Author  : Adolf
# @Site    : 
# @File    : parse_original_data.py
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


read_docx_file(docx_path="DocumentReview/DocData/Sample/安保服务外包合同-甲方.docx")
