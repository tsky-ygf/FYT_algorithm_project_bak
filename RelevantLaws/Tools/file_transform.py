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

file = st.file_uploader('上传文件', type=['docx'], key=None)


@st.cache
def get_data(_file):
    _docx_content_list = []

    if _file is not None:
        document = Document(_file)
        for index, paragraph in enumerate(document.paragraphs):
            one_text = paragraph.text.replace(" ", "").replace("\u3000", "")
            if one_text != "":
                # st.markdown(one_text)
                _docx_content_list.append(one_text)

    return _docx_content_list


# transform docx to csv
def transform_data(_docx_content_list):
    part = []
    chapter = []
    clause = []

    current_part = "第一编"
    current_chapter = "第一章"
    current_subsection = "第一节"
    current_clause = "第一条"

    for one_law in _docx_content_list:
        print(one_law)
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


docx_content_list = get_data(file)
df = transform_data(docx_content_list)
st.write(df)

st.download_button(
     label="Download data as CSV",
     data=df.to_csv().encode('utf-8'),
     file_name='laws.csv',
     mime='text/csv',
 )
