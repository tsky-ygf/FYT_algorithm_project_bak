#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 13:27
# @Author  : Adolf
# @Site    : 
# @File    : st_show.py
# @Software: PyCharm
import streamlit as st
from DocumentReview.ContractReview.loan_acknowledgement import loan_acknowledgement
from DocumentReview.ParseFile.parse_word import read_docx_file

contract_type = st.sidebar.selectbox("请选择合同类型", ["借条"], key="借条")
mode_type = st.sidebar.selectbox("请选择上传数据格式", ["text", "docx"], key="text")


@st.cache
def get_data(_file):
    _text = read_docx_file(_file)
    # print(_text)
    return "\n".join(_text)


def show_res(_text):
    loan_acknowledgement.review_main(content=_text, mode="text")
    st.write("合同审核结果:")
    st.write(loan_acknowledgement.review_result)


if mode_type == "text":
    text = st.text_area(label="请输入文本内容", height=600,
                        value="借 条\n为购买房产，今收到好友张三（身份证号）以转账方式出借的人民币壹万元整（￥10000.00元），\
         借期拾个月，月利率1%，于2023年05月23日到期时还本付息。逾期未还，则按当期一年期贷款市场报价利率（LPR）的4倍计付逾期利息。\n如任何一方（借款人、债务人）\
         违约，守约方（出借人、债权人）为维护权益向违约方追偿的一切费用（包括但不限于律师费、诉讼费、保全费、交通费、差旅费、鉴定费等等）均由违约方承担。\n身份证载明\
         的双方（各方）通讯地址可作为送达催款函、对账单、法院送达诉讼文书的地址，因载明的地址有误或未及时告知变更后的地址，\
         导致相关文书及诉讼文书未能实际被接收的、邮寄送达的，相关文书及诉讼文书退回之日即视为送达之日。\n借款人的微信号为：ffdsaf\n借款人：李四\n身份\
         证号：123132423142314231\n联系电话：13242314123\n借款人：李四媳妇\n身份证号：12343124\n联系电话：1342324123\n家庭住\
         址：（具体到门牌号）\n2022年05月12日", key="text")

    # show_res()


elif mode_type == "docx":
    file = st.file_uploader('上传文件', type=['docx'], key=None)

    text = get_data(file)
    # loan_acknowledgement.review_main(content=file_path, mode="docx")


else:
    raise Exception("暂不支持该格式")

run = st.button("开始审核合同ing", key="run")

if run:
    show_res(text)
