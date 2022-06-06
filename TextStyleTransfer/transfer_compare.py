#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 15:35
# @Author  : Adolf
# @Site    : 
# @File    : transfer_compare.py
# @Software: PyCharm
import streamlit as st
from googletrans import Translator
import requests

translator = Translator(service_urls=[
    'translate.google.cn'
])

text = st.text_area(label="文字输入", height=None, value="你好世界", key="text")
run = st.button('一键智能转写')
if run:
    en_res1 = translator.translate(text, dest='en').text
    ch_res1 = translator.translate(en_res1, dest='zh-CN').text

    st.write(ch_res1)
    st.write('-' * 50 + '分割线' + '-' * 50)

    # text2 = st.text_area(label="智能转写模型2", height=None, value="你好世界", key="text")

    en_res2 = translator.translate(text, dest='ja').text
    ch_res2 = translator.translate(en_res2, dest='zh-CN').text

    st.write(ch_res2)
    st.write('-' * 50 + '分割线' + '-' * 50)

    # model.to("cuda:3")
    # text3 = st.text_area(label="智能转写模型3", height=None, value="你好世界", key="text")

    url = "http://172.19.82.199:7999/translation"
    r = requests.post(url, json={"content": text})
    ch_res3 = r.json()['result']
    # print(en_res3)

    st.write(ch_res3)
    st.write('-' * 50 + '分割线' + '-' * 50)
