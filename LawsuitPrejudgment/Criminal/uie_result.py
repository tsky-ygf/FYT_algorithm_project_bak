#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 10:16
# @Author  : Adolf
# @Site    : 
# @File    : uie_result.py
# @Software: PyCharm
import streamlit as st
# import pandas as pd
from paddlenlp import Taskflow

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_schema = st.text_input("请输入文本：")
text = st.text_area(label="请输入文本内容", height=300, value="", key="text")
ie = Taskflow('information_extraction', schema=use_schema, device_id=2)
run = st.button("抽取")
if run:
    res = ie(text)
    # print(res)
    find_schema = []
    # print(res)
    st.write(res)
    # for key, values in res[0].items():
    #     st.write(index + 1, '.', value['text'])
