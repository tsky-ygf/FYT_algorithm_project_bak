#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 09:43
# @Author  : Adolf
# @Site    : 
# @File    : contract_for_server.py
# @Software: PyCharm
import json
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
                                                                      device="cpu")
    return acknowledgement_dict
