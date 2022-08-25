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
# from paddlenlp import Taskflow
# import pycorrector
from loguru import logger
from pprint import pprint
from pypinyin import pinyin, lazy_pinyin


# from annotated_text import annotated_text


# text_correction = Taskflow("text_correction")


@st.cache
def get_data(_file):
    _text = read_docx_file(_file)
    # print(_text)
    return "\n".join(_text)


contract_type = st.sidebar.selectbox("请选择合同类型",
                                     ["借条", "借款", "劳动", '房屋租赁', '买卖', '劳务', '保密', '采购', '一般租赁'],
                                     key="合同类型")
mode_type = st.sidebar.selectbox("请选择上传数据格式", ["docx", "文本", "txt"], key="text")
usr = st.sidebar.selectbox("请选择立场", ['甲方', '乙方'], key="中立方")

contract_type = ''.join(lazy_pinyin(contract_type))
config_path = "DocumentReview/Config/{}.csv".format(contract_type)
model_path = "model/uie_model/new/{}/model_best/".format(contract_type)
# model_path = "model/uie_model/export_cpu/{}/inference".format(contract_type)
# print(contract_type)

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

acknowledgement = BasicUIEAcknowledgement(config_path=config_path,
                                          model_path=model_path,
                                          device="1")
correct = st.button("文本纠错")
run = st.button("开始审核")

# use paddleNLP to correct the text
# if correct:
#     res_correct = text_correction(text)
#     if len(res_correct) > 0:
#         # st.write(annotated_text(res_correct))
#         # st.write("纠错后的文本")
#         # st.write(text)
#         print(res_correct)
#         res_text = []
#         last_index = 0
#         for one_error in res_correct[0]['errors']:
#             one_position = one_error['position']
#             print(one_position)
#             res_text.append(text[last_index:one_position])
#             res_text.append((text[one_position], one_error['correction'][text[one_position]], '#FF8B72'))
#             # res_text.append(text[one_position + 1:])
#             last_index = one_position + 1
#         res_text.append(text[last_index:])
#         annotated_text(*res_text)
#     else:
#         st.write("错别字审核通过")
if correct:
    import requests
    import pandas as pd

    # st.write(text)
    r = requests.post("http://172.19.82.199:6598/macbert_correct", json={"text": text})
    result = r.json()
    if result['success']:
        result = result['result']
        # if len(result) > 0:
        #     st.write(result)
        # st.write(result)
        origin_text_list = []
        correct_text_list = []
        error_list = []
        for one_error in result:
            if len(one_error[2]) == 0:
                # st.write("{} ====> 纠错通过".format(one_error[0]))
                pass
            else:
                # st.write("{} ====> {}".format(one_error[0], one_error[1]))
                # res_dict[one_error[0]] = one_error[1]
                origin_text_list.append(one_error[0])
                correct_text_list.append(one_error[1])
                error_text = ""
                # error_list.append(one_error[2][0])
                for er in one_error[2]:
                    error_text += er[0] + "===>" + er[1] + "\n"
                st.write(error_text)
                error_list.append(error_text)
                # for
        res_dict = {"原始文本": origin_text_list, "纠错后的文本": correct_text_list, "错别字": error_list}
        print(res_dict)

        my_df = pd.DataFrame.from_dict(res_dict)
        st.table(my_df)
        # my_df.columns = ['纠错后的句子']
        # st.dataframe(my_df)

    else:
        logger.error(result['error_msg'])
        # result = []

if run:
    acknowledgement.review_main(content=text, mode="text", usr=usr)
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
            if "法律建议" in value and value["法律建议"] != "":
                st.markdown("法律建议：{}".format(value['法律建议']))
            if "风险点" in value and value["风险点"] != "":
                st.markdown("风险点：{}".format(value['风险点']))
            if "法律依据" in value and value["法律依据"] != "":
                st.markdown("法律依据：{}".format(value['法律依据']))
            if "风险等级" in value and value["风险等级"] != "":
                st.markdown("风险等级：{}".format(value['风险等级']))
        except Exception as e:
            print(e)
            st.write("这个审核点小朱正在赶制，客官请稍等。。。。可爱")
