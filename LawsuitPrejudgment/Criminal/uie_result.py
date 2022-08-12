#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 10:16
# @Author  : Adolf
# @Site    : 
# @File    : uie_result.py
# @Software: PyCharm
import streamlit as st
from LawsuitPrejudgment.Criminal.feature_extraction import get_xing7_result

text = st.text_area(value="诸暨市人民检察院指控：2012年10月25日1时许，被告人郑小明伙同彭小清窜到诸暨市暨阳街道跨湖路99号永"
                          "鑫花园，撬断围栏进入小区，由被告人郑小明望风，彭小清通过下水道进入小区20幢5单元被害人杨燕家，窃得"
                          "部分财物，后两人翻墙逃跑。被告人郑小明在逃跑途中被民警抓获。被告人郑小明于2009年5月21日因犯盗窃"
                          "罪被江西省抚州市临川区人民法院判处有期徒刑4年6个月，于2012年9月7日刑满释放。被告人郑小明对公诉机关指控"
                          "的事实和罪名及证据均无异议，未提出辩解。", height=300, label="请输入裁判文书内容", key="text")
run = st.button("抽取")
if run:
    res = get_xing7_result(text)[0]
    # st.write(res)
    for key, value in res.items():
        st.markdown(f'## {key}')
        for content in value:
            st.markdown(f'- {content["text"]}')
        st.write('-' * 50 + '分割线' + '-' * 50)
