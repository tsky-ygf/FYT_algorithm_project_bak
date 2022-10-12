#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 09:43
# @Author  : Adolf
# @Site    : 
# @File    : contract_for_server.py
# @Software: PyCharm
import argparse
import json
import os
import re
import time
import uuid
from dataclasses import dataclass

from docx import Document

from DocumentReview.src.ParseFile import read_txt_file, read_docx_file
from DocumentReview.src.common_contract import BasicPBAcknowledgement

CONTRACT_SERVER_DATA_PATH = "DocumentReview/Config/schema/contract_server_data.json"


def get_support_contract_types():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("support_contract_types")


def get_contract_type_list():
    support_contract_types = get_support_contract_types()
    return [item.get("type_id") for item in support_contract_types]


def get_user_standpoint():
    with open(CONTRACT_SERVER_DATA_PATH, "r", encoding="utf-8") as f:
        user_standpoints = json.load(f).get("user_standpoints")
    return user_standpoints


@dataclass
class CommonModelArgs:
    model_load_path = "model/PointerBert/PBert1011_common_all_20sche_aug.pt"
    model = "model/language_model/chinese-roberta-wwm-ext"
    common_schema_path = "DocumentReview/Config/config_common.csv"
    bert_emb_size = 768
    hidden_size = 200


def init_model():
    common_model_args = CommonModelArgs()
    print('=' * 50, '模型初始化...', '=' * 50)
    print(time.localtime())
    acknowledgement = BasicPBAcknowledgement(contract_type_list=get_contract_type_list(),
                                             config_path_format="DocumentReview/Config/schema/{}.csv",
                                             model_path_format="model/uie_model/export_cpu/{}/inference",
                                             common_model_args=common_model_args,
                                             log_level="INFO",
                                             device="cpu")
    return acknowledgement


def get_text_from_file_link_path(file_link):
    os.system('cd data/uploads && wget ' + file_link)
    filename = file_link.split('/')[-1]
    if '.docx' in file_link:
        data = read_docx_file(os.path.join('data/uploads', filename))
    elif '.txt' in file_link:
        data = read_txt_file(os.path.join('data/uploads', filename))
    else:
        data = "invalid input"
    os.remove(os.path.join('data/uploads', filename))
    return data


def get_text_from_file(file):
    filename = file.filename
    if filename.rsplit('.', 1)[1].lower() == 'txt':
        t = uuid.uuid3(uuid.NAMESPACE_DNS, filename)
        filename = str(t) + '.txt'
        file.save(os.path.join('data/uploads', filename))
        data = read_txt_file(os.path.join('data/uploads', filename))
        os.remove(os.path.join('data/uploads', filename))
    elif filename.rsplit('.', 1)[1].lower() == 'docx':
        t = uuid.uuid3(uuid.NAMESPACE_DNS, filename)
        filename = str(t) + '.docx'
        file.save(os.path.join('data/uploads', filename))
        data = read_docx_file(os.path.join('data/uploads', filename))
        os.remove(os.path.join('data/uploads', filename))
    else:
        data = "invalid input"

    return data
