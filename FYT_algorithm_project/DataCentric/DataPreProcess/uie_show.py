#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 09:08
# @Author  : Adolf
# @Site    : 
# @File    : uie_show.py
# @Software: PyCharm
import streamlit as st
from paddlenlp import Taskflow
from pprint import pprint

st.title("通用信息抽取工具")
text = st.text_area(label="需要提取的文字内容",
                    value="",
                    key=None)

schema = st.text_input(label="schema",)

if schema and text:
    # schema = ['标题', '借款人', '借款金额', '借款期限', '借款利率', '还款方式', '还款日期', '逾期利率', '逾期费用']
    ie = Taskflow('information_extraction', schema=[schema], device_id=1)
    res = ie(text)
    pprint(res)
    st.write(res)
