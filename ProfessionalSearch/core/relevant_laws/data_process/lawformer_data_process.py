#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 10:12
# @Author  : Adolf
# @Site    :
# @File    : lawformer_data_process.py
# @Software: PyCharm
import hanlp
import json
from ProfessionalSearch.core.relevant_laws.data_process.data_process import get_fileter_data
from Utils.logger import print_run_time
from tqdm import tqdm

# from pathos.multiprocessing import ProcessingPool as Pool
# import torch.multiprocessing as mp

# import multiprocessing


# mp.set_start_method("spawn")


def handle_one_fact(_item, _ner):
    """
    处理一条法条
    :param _ner: 调用的ner模型
    :param _item: 每一条具体的内容
    :return:
    """
    fact = _item["fact"]
    try:
        fact = get_fileter_data(fact, _ner)
    except:
        # fact = ""
        pass
    # time.sleep(0.02)
    _item["fact"] = fact

    return _item


@print_run_time
def get_law_data(law_file, law_save_file, _ner):
    """
    获取法律数据
    :param _ner:
    :param law_save_file:
    :param law_file:
    :return:
    """
    with open(law_file, "rb") as f:
        load_list = json.load(f)

    sentences = []

    # def handle_data(_sentences):
    #     for item in load_list:
    #         _sentences.append(handle_one_fact(item))

    sentences = []
    for item in tqdm(load_list):
        sentences.append(handle_one_fact(item, _ner))

    # with Pool(20) as p:
    #     sentences = list(tqdm(p.imap(handle_one_fact, load_list), total=len(load_list), desc="处理法条"))

    # pool = multiprocessing.Pool(processes=8)

    # for item in tqdm(load_list):
    # pool.imap(handle_one_fact, item)

    # with mp.Pool(processes=8) as pool:
    #     sentences = list(
    #         tqdm(pool.imap(handle_one_fact, (load_list, [_ner] * len(load_list),)), total=len(load_list), desc="处理法条"))
    #
    # pool.close()
    # pool.join()

    with open(law_save_file, "w") as f:
        json.dump(sentences, f, ensure_ascii=False)


if __name__ == "__main__":
    ner = hanlp.load(
        save_dir=hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH,
        devices=0,
    )
    train_json_path = "data/fyt_train_use_data/CAIL-Long/civil/train.json"
    dev_json_path = "data/fyt_train_use_data/CAIL-Long/civil/dev.json"
    test_json_path = "data/fyt_train_use_data/CAIL-Long/civil/test.json"

    train_save_path = "data/fyt_train_use_data/CAIL-Long/civil/train_filter.json"
    dev_save_path = "data/fyt_train_use_data/CAIL-Long/civil/dev_filter.json"
    test_save_path = "data/fyt_train_use_data/CAIL-Long/civil/test_filter.json"

    get_law_data(train_json_path, train_save_path, ner)
    get_law_data(dev_json_path, dev_save_path, ner)
    get_law_data(test_json_path, test_save_path, ner)
