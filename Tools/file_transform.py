#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 09:40
# @Author  : Adolf
# @Site    : 
# @File    : file_transform.py
# @Software: PyCharm
import streamlit as st
import pandas as pd
import numpy as np
from docx import Document
import re

st.title('File Transform')

st.write('This is a file transform tool')
st.write('upload your docx file')

file_type = st.selectbox('选择文件类型', ['带有编', '带有章', '只有条目'])
file = st.file_uploader('上传文件', type=['docx'], key=None)


@st.cache
def get_data(_file):
    _docx_content_list = []

    if _file is not None:
        document = Document(_file)
        for index, paragraph in enumerate(document.paragraphs):
            one_text = paragraph.text.replace(" ", "").replace("\u3000", "")
            one_text_list = one_text.split('\n')
            # if one_text != "":
            one_text_list = [x for x in one_text_list if x != ""]
            _docx_content_list += one_text_list

    return _docx_content_list


# transform docx to csv
def transform_data_1(_docx_content_list):
    part = []
    chapter = []
    clause = []

    current_part = "第一编"
    current_chapter = "第一章"
    current_subsection = "第一节"
    current_clause = "第一条"

    for one_law in _docx_content_list:
        # print(one_law)
        if len(re.findall("第.*编", one_law[:5])) > 0:
            current_part = one_law
        elif len(re.findall("第.*章", one_law[:5])) > 0:
            current_chapter = one_law
        elif len(re.findall("第.*节", one_law[:5])) > 0:
            current_subsection = one_law
        elif len(re.findall("第.*条", one_law[:12])) > 0:
            # current_part = one_law
            part.append(current_part)
            chapter.append(current_chapter)
            clause.append(current_clause)
            current_clause = one_law
        else:
            current_clause += one_law

    res_dict = {"part": part,
                "chapter": chapter,
                "clause": clause}
    res_df = pd.DataFrame(res_dict)
    res_df = res_df[1:]
    return res_df


def transform_data_2(_docx_content_list):
    chapter = []
    clause = []

    current_chapter = "第一章"
    current_clause = "第一条"

    for one_law in _docx_content_list:
        print(one_law)
        if len(re.findall("第.*章", one_law[:5])) > 0:
            current_chapter = one_law
        elif len(re.findall("第.*条", one_law[:12])) > 0:
            chapter.append(current_chapter)
            clause.append(current_clause)
            current_clause = one_law
        else:
            current_clause += one_law

    res_dict = {"chapter": chapter,
                "clause": clause}
    res_df = pd.DataFrame(res_dict)
    res_df = res_df[1:]
    print(res_df)
    return res_df


def transform_data_3(_docx_content_list):
    clause = []
    current_clause = "第一条"

    for one_law in _docx_content_list:
        # print(one_law)
        if len(re.findall("第.*条", one_law[:12])) > 0:
            one = re.findall("第.*条", current_clause[:12])[0]
            # print(one)
            clause.append(current_clause.replace(one, one + "_"))
            current_clause = one_law
        else:
            current_clause += one_law
        # break
    res_dict = {"clause_content": clause}
    res_df = pd.DataFrame(res_dict)
    res_df = res_df[1:].applymap(lambda x: x.replace(" ", ""))
    # 如果需要实现分列效果，可以通过expand=True参数返回DataFrame
    res_df = res_df["clause_content"].str.split("_", expand=True)
    res_df.columns = ["clause", "content"]
    # print(res_df)
    return res_df


if file_type == "带有编":
    transform_data = transform_data_1  # 带有编
elif file_type == "带有章":
    transform_data = transform_data_2  # 带有章
elif file_type == "只有条目":
    transform_data = transform_data_3  # 只有条目
else:
    raise Exception("file_type error")  # 其他情况

docx_content_list = get_data(file)
df = transform_data(docx_content_list)
st.write(df)

st.download_button(
    label="Download data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='laws.csv',
    mime='text/csv',
)
