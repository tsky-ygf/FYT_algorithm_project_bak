#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 09:08
# @Author  : Adolf
# @Site    : 
# @File    : uie_show.py
# @Software: PyCharm
import streamlit as st
from pprint import pprint
from paddlenlp import Taskflow

st.title("通用信息抽取工具")
text = st.text_area(label="需要提取的文字内容",
                    value="""为购买房产，今收到好友张三（身份证号）以转账方式出借的人民币壹万元整（￥10000.00元），借期拾个月，月利率1%，于××××年××月××日到期时还本付息。逾期未还，则按当期一年期贷款市场报价利率（LPR）的4倍计付逾期利息。""",
                    key=None)

schema = st.text_input(label="schema", value="借款人")

# schema = ['标题', '借款人', '借款金额', '借款期限', '借款利率', '还款方式', '还款日期', '逾期利率', '逾期费用']
ie = Taskflow('information_extraction', schema=[schema], device_id=1)
# pprint(ie(text))
st.write(ie(text))
