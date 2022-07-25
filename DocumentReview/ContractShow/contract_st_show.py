#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/21 13:23
# @Author  : Adolf
# @Site    : 
# @File    : contract_st_show.py
# @Software: PyCharm
import streamlit as st
from DocumentReview.ParseFile.parse_word import read_docx_file
from DocumentReview.ContractReview.basic_contract import BasicUIEAcknowledgement
from paddlenlp import Taskflow
# import pycorrector

from pprint import pprint
from annotated_text import annotated_text

text_correction = Taskflow("text_correction")


@st.cache
def get_data(_file):
    _text = read_docx_file(_file)
    # print(_text)
    return "\n".join(_text)


contract_type = st.sidebar.selectbox("请选择合同类型", ["借条", "借款合同", "劳动合同", '租房合同', '买卖合同', '劳务合同'], key="合同类型")
mode_type = st.sidebar.selectbox("请选择上传数据格式", ["docx", "文本", "txt"], key="text")
usr = st.sidebar.selectbox("请选择立场", ['甲方', '乙方'], key="中立方")

if mode_type == "docx":
    file = st.file_uploader('上传文件', type=['docx'], key=None)
    text = get_data(file)
elif mode_type == "txt":
    text = st.text_area(label='请输入文本', value='', height=600, key=None)
elif mode_type == "文本":
    text = st.text_area(label='请输入文本', value='', height=600, key=None)
else:
    raise Exception("暂时不支持该数据格式")

if usr == '甲方':
    usr = 'Part A'
else:
    usr = 'Part B'

if contract_type == "借条":
    config_path = "DocumentReview/Config/jietiao.csv"
    model_path = "model/uie_model/model_best/"

elif contract_type == "借款合同":
    config_path = "DocumentReview/Config/jiekuan.csv"
    model_path = "model/uie_model/jkht/model_best/"
elif contract_type == "劳动合同":
    config_path = "DocumentReview/Config/labor.csv"
    model_path = "model/uie_model/labor/model_best/"
elif contract_type == "租房合同":
    config_path = "DocumentReview/Config/fangwuzulin.csv"
    model_path = "model/uie_model/fwzl/model_best/"
elif contract_type == "买卖合同":
    config_path = "DocumentReview/Config/maimai.csv"
    model_path = "model/uie_model/maimai/model_best/"
elif contract_type == "劳务合同":
    config_path = "DocumentReview/Config/laowu.csv"
    model_path = 'model/uie_model/guyong/model_best/'
else:
    raise Exception("暂时不支持该合同类型")

acknowledgement = BasicUIEAcknowledgement(config_path=config_path,
                                          model_path=model_path,
                                          usr=usr)
correct = st.button("文本纠错")
run = st.button("开始审核")
if correct:
    res_correct = text_correction(text)
    if len(res_correct) > 0:
        # st.write(annotated_text(res_correct))
        # st.write("纠错后的文本")
        # st.write(text)
        print(res_correct)
        res_text = []
        last_index = 0
        for one_error in res_correct[0]['errors']:
            one_position = one_error['position']
            print(one_position)
            res_text.append(text[last_index:one_position])
            res_text.append((text[one_position], one_error['correction'][text[one_position]], '#FF8B72'))
            # res_text.append(text[one_position + 1:])
            last_index = one_position + 1
        res_text.append(text[last_index:])
        annotated_text(*res_text)
    else:
        st.write("错别字审核通过")
if run:
    # corrected_sent, detail = pycorrector.correct(text)
    # print(corrected_sent, detail)

    acknowledgement.review_main(content=text, mode="text")
    pprint(acknowledgement.review_result, sort_dicts=False)
    index = 1
    for key, value in acknowledgement.review_result.items():
        # st.write(key, value)
        pprint(value)
        st.markdown('### {}、审核点：{}'.format(index, key))
        index += 1
        try:
            if "审核结果" in value and value["审核结果"] != "":
                st.markdown("审核结果：{}".format(value['审核结果']))

            if value['审核结果'] == "通过" and value["风险等级"] == "低":
                continue

            if "内容" in value and value["内容"] != "":
                st.markdown("审核内容：{}".format(value['内容']))
            if "法律依据" in value and value["法律依据"] != "":
                st.markdown("法律依据：{}".format(value['法律依据']))
            if "风险等级" in value and value["风险等级"] != "":
                st.markdown("风险等级：{}".format(value['风险等级']))
        except Exception as e:
            print(e)
            st.write("这个审核点小朱正在赶制，客官请稍等。。。。可爱")
