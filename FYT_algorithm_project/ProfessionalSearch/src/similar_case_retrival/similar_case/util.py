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
    word_idx_path = "data/search/textCNN/vector/pad_word_index.json"
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


def desensitization(text: str) -> list:
    # 脱敏人名, 如果陈重姓，使用陈某1，陈某2
    res = ""
    # 抽取人名
    name_list = extract_name_LTP(text)
    # 替换人名
    res = replace_name(name_list, text)
    # 脱敏身份证号

    return res


def extract_name_hanlp(text: str) -> list:
    from pyhanlp import HanLP

    if not text:
        return []
    segment = HanLP.newSegment().enableNameRecognize(True)
    user_list = []
    for i in segment.seg(text):
        split_words = str(i).split("/")
        word, tag = split_words[0], split_words[-1]
        if tag == "nr":
            user_list.append(word)
    return user_list


def extract_name_LTP(text: str) -> list:
    from ltp import LTP

    ltp = LTP()  # 默认加载 Small 模型，下载的路径是：~/.cache/torch/ltp
    cws, pos, ner = ltp.pipeline([text], tasks=["cws", "pos", "ner"]).to_tuple()
    print("ner", ner)
    nh_user_list = []
    for ner_i in ner[0]:
        if ner_i[0] == "Nh":
            nh_user_list.append(ner_i[1])
    return nh_user_list


def replace_name(name_list: list, text: str) -> str:
    from pypinyin import pinyin, Style

    # name分组
    name_first = ""
    name_list_par = []
    name_list_temp = []
    name_rem_list = list(set(name_list))
    # 名字字典排序
    name_rem_list.sort(key=lambda keys: [pinyin(i, style=Style.TONE3) for i in keys])
    print(name_rem_list)
    # 依同姓分组
    for index, name_item in enumerate(name_rem_list):
        if name_first:
            if name_first[0] == name_item[0]:
                name_list_temp.append(name_item)
                if index == len(name_rem_list) - 1:
                    name_list_par.append(name_list_temp)
            else:
                name_list_par.append(name_list_temp)
                name_list_temp = []
                name_first = name_item[0]
                name_list_temp.append(name_item)
                if index == len(name_rem_list) - 1:
                    name_list_par.append(name_list_temp)
        else:
            name_first = name_item
            name_list_temp.append(name_first)
            if len(name_rem_list) == 1:
                name_list_par.append(name_list_temp)
    # 依字典序分组替换
    print(name_list_par)
    for name_i_list in name_list_par:
        same_index = 0
        if len(name_i_list) > 1:
            for name_i_item in name_i_list:
                same_index += 1
                name_new = name_i_item[0] + "某" + str(same_index)
                text = text.replace(name_i_item, name_new)
        else:
            text = text.replace(name_i_list[0], name_i_list[0][0] + "某")
    return text


if __name__ == "__main__":
    # get_vector_wiki()
    text1 = "《八佰》（英語：The Eight Hundred）是一部于2020年上映的以中国历史上的战争为题材的电影，由管虎执导，黄志忠、黄骏豪、张俊一、张一山....."
    text = (
        "陕西省宝鸡市渭滨区人民法院民事裁定书（2020）陕0302民初1935号原告：麻文彬，男，汉族，1988年9月1日出生，公民身份号码XXXXXXXXXXXXXXXXXX"
        "，住宝鸡市渭滨区被告：陕西中国旅行社有限责任公司，住所地陕西省西安市碑林区，统一社会信用代码91610000727330027M"
        "。法定代表人：高举，该公司董事长兼总经理。原告麻文彬与被告陕西中国旅行社有限责任公司旅游合同纠纷一案，本院于2020年5月6日立案。原告诉称，2020年1月26"
        "日，原、被告通过网络签订了《大陆居民出境游合同》，约定于2020年1月29日出发去越南芽庄旅行。合同签订后，原告向被告支付12000元。2020年1月26"
        "日中午，原告接到被告旅行延期的通知，因原告工作原因，旅行无法改期，故要求被告退款，遭到拒绝。为维护原告合法权益，现诉至法院。被告陕西中国旅行社有限责任公司在提交答辩状期间，对管辖权提出异议。根据《中华人民共和国民事诉讼法》第二十三条之规定，被告认为本案渭滨区人民法院没有管辖权。本院经审查认为，依照《中华人民共和国民事诉讼法》第二十三条的规定，因合同纠纷提起的诉讼，由被告住所地或者合同履行地人民法院管辖。本案是旅游合同纠纷，合同履行地为越南芽庄，不在本国内，因此，只能由被告住所地人民法院管辖，被告住所地为西安市碑林区，故本院对本案无管辖权。依照《中华人民共和国民事诉讼法》第二十三条，第三十六条，《最高人民法院关于适用中华人民共和国民事诉讼法>的解释》第一百二十七条第一款之规定，裁定如下：被告陕西中国旅行社有限责任公司提出的管辖权异议成立，本案移送西安市碑林区人民法院处理。如不服本裁定，可以在裁定书送达之日起十日内，向本院递交上诉状，并按对方当事人或者代表人的人数提出副本，上诉于陕西省宝鸡市中级人民法院。审判员李鸿衍二○二〇年五月二十一日法官助理杜岩书记员王璐1 "
    )
    print(desensitization(text1))
    # test(text)
