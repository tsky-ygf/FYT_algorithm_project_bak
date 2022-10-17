import json
import logging
from typing import List, Dict
import pymysql
import torch
import jieba

from Utils import print_run_time


def get_civil_law_documents_by_id_list(id_list: List[str], table_name) -> List[Dict]:
    # 打开数据库连接
    db = pymysql.connect(
        host="172.19.82.227",
        user="root",
        password="Nblh@2022",
        database="judgments_data",
    )

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    try:
        format_strings = ",".join(["%s"] * len(id_list))
        # 执行SQL语句
        cursor.execute(
            "SELECT uq_id, jslcm FROM "
            + table_name
            + " WHERE uq_id in (%s)" % format_strings,
            tuple(id_list),
        )
        # 获取所有记录列表
        fetched_data = cursor.fetchall()
        law_documents = [{"uq_id": row[0], "jslcm": row[1]} for row in fetched_data]
    except:
        logging.error("Error: unable to fetch data")
        law_documents = []
    # 关闭数据库连接
    db.close()
    return law_documents


@print_run_time
def get_vector_wiki():
    from gensim.models import KeyedVectors

    vector_path = "data/bxh_search_data/textCNN/vector/sgns.wiki2.bigram-char"
    word_idx_path = "data/bxh_search_data/textCNN/vector/pad_word_index.json"
    weight_save_path = "data/bxh_search_data/textCNN/vector/wiki_vec_pad.pt"
    wvmodel = KeyedVectors.load_word2vec_format(vector_path)
    vocab_size = 352272 + 1  # len(vocab)
    embed_size = 300
    weight = torch.zeros(vocab_size, embed_size)
    with open(word_idx_path, "r") as f:
        word_to_idx = json.load(f)
        for i in range(len(wvmodel.index2word)):
            try:
                index = word_to_idx[wvmodel.index2word[i]]
            except:
                continue
            weight[index, :] = torch.from_numpy(
                wvmodel.get_vector(wvmodel.index2word[i])
            )
        f.close()
    torch.save(weight, weight_save_path)
    return weight


@print_run_time
def get_index_list(text):
    # 分词
    text_list = jieba.lcut(text)
    # padding
    if len(text_list) < 4:
        for i in range(5 - len(text_list)):
            text_list.append("<PAD>")
    # 获取index并合并为list
    word_idx_path = "data/bxh_search_data/textCNN/vector/pad_word_index.json"
    with open(word_idx_path, "r") as f:
        word_to_idx = json.load(f)
        index_text = []
        for item_text in text_list:
            try:
                index = word_to_idx[item_text]
            except:
                continue  # 简单地过滤了OOV词汇，待优化
            index_text.append(index)
    return index_text


if __name__ == "__main__":
    get_vector_wiki()
    pass
