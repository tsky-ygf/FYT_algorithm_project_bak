#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 09:43
# @Author  : Adolf
# @Site    : 
# @File    : contract_for_server.py
# @Software: PyCharm
import json
import os
import re

from docx import Document

from DocumentReview.ContractReview.showing_sample import BasicUIEAcknowledgement

CONTRACT_SERVER_DATA_PATH = "DocumentReview/Config/contract_server_data.json"


def get_support_contract_types():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("support_contract_types")


def get_user_standpoint():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        user_standpoints = json.load(f).get("user_standpoints")
    return user_standpoints


def get_contract_type_list():
    support_contract_types = get_support_contract_types()
    return [item.get("type_id") for item in support_contract_types]


def init_model():
    contract_type_list = get_contract_type_list()
    acknowledgement_dict = {}
    for contract_type in contract_type_list:
        config_path = "DocumentReview/Config/{}.csv".format(contract_type)
        model_path = "model/uie_model/export_cpu/{}/inference".format(contract_type)
        acknowledgement_dict[contract_type] = BasicUIEAcknowledgement(config_path=config_path,
                                                                      model_path=model_path,
                                                                      device="cpu",
                                                                      logger_file='log/contract_review/model.log')
    return acknowledgement_dict



def file_link_path_to_text(file_link):
    os.system('cd data/uploads && wget ' + file_link)
    if '.docx' in file_link:
        data = read_docx_file(os.path.join('data/uploads', file_link))
    elif '.txt' in file_link:
        data = read_txt_file(os.path.join('data/uploads', file_link))
    else:
        data =  "invalid input"


# 读取docx 文件
def read_docx_file(docx_path):
    document = Document(docx_path)
    # tables = document.tables
    all_paragraphs = document.paragraphs
    return_text_list = []
    for index, paragraph in enumerate(all_paragraphs):
        one_text = paragraph.text.replace(" ", "").replace("\u3000", "")
        if one_text != "":
            return_text_list.append(one_text)
    # print(return_text_list)
    data = '\n'.join(return_text_list)
    data = data.replace('⾄', '至').replace('中华⼈民', '中华人民') \
        .replace(' ', ' ').replace(u'\xa0', ' ').replace('\r\n', '\n')
    data = re.sub("[＿_]+", "", data)
    return data