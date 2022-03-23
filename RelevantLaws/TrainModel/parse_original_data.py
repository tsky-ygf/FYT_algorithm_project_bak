#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:04
# @Author  : Adolf
# @Site    : 
# @File    : parse_original_data.py
# @Software: PyCharm
import pandas as pd
from docx import Document
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# 读取docx 文件
def parse_docx_to_csv(origin_doc):
    document = Document(origin_doc)
    all_paragraphs = document.paragraphs

    parts = []
    chapters = []
    items = []
    contents = []

    current_part = ""
    current_chapter = ""

    for index, paragraph in enumerate(all_paragraphs):
        # one_text = paragraph.text.replace(" ", "").replace("\u3000", "")
        one_text = paragraph.text.strip().replace('\u3000', " ")
        # print(one_text)
        if len(re.findall('第*编', one_text[:3])) > 0:
            current_part = one_text
            # print(current_part)
        if len(re.findall('第*章', one_text[:3])) > 0:
            current_chapter = one_text

        if len(re.findall('第*条', one_text[:3])) > 0:
            parts.append(current_part)
            chapters.append(current_chapter)
            items.append(one_text)
            contents.append(" ")
        # break

    res_dict = {"总编": parts,
                "章节": chapters,
                "条目": items,
                "内容": contents}
    res_df = pd.DataFrame(res_dict)
    # print(res_df)
    res_df.to_csv("RelevantLaws/LawsData/民法典.csv", index=False)


parse_docx_to_csv(origin_doc="RelevantLaws/LawsData/民法典.docx")
