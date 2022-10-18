#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 13:07
# @Author  : Adolf
# @Site    :
# @File    : data_process.py
# @Software: PyCharm
import hanlp
import re
import json

test_json_path = "data/fyt_train_use_data/CAIL-Long/civil/test.json"
with open(test_json_path, "rb") as f:
    test_load_list = json.load(f)


def get_fileter_data(_fact, _ner, is_tokenization=False):
    """
    对输入的事实部分进行数据预处理
    :param _fact: 事实部分
    :return:
    """
    fact_list = re.split("[：|；|，|。]", _fact)
    fact_list = list(map(lambda x: re.sub("[^\u4e00-\u9fa5]+", "", x), fact_list))
    _fact = ",".join(fact_list)
    # print(_fact)

    resp = _ner(_fact, tasks="ner/msra")
    if is_tokenization:
        # print(resp["tok/fine"])
        _fact = " ".join(resp["tok/fine"])

    for one in resp["ner/msra"]:
        _fact = _fact.replace(one[0], "")

    _fact = _fact.replace("、", "").replace(",", "").replace("  ", " ")
    # print(_fact)
    return _fact


if __name__ == "__main__":
    ner = hanlp.load(
        hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
    )
    get_fileter_data(test_load_list[10]["fact"], _ner=ner, is_tokenization=True)
