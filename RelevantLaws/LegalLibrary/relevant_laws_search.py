#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 13:41
# @Author  : Adolf
# @Site    : 
# @File    : relevant_laws_search.py
# @Software: PyCharm
import jieba
from RelevantLaws.LegalLibrary.read_legal_from_db import search_data_from_es


def get_law_search_result(text="", sxx_list=None, legal_list=None, size=10):
    """
    法条搜索
    :param text: 搜索文本
    :param sxx_list: 时效性列表
    :param legal_list: 法条种类列表
    :param size: 搜索结果数量
    :return:
    """
    if sxx_list is None:
        sxx_list = ['有效', '已修改', '尚未生效', '已废止']
    if legal_list is None:
        legal_list = ['宪法', '法律', '行政法规', '监察法规', '司法解释', '地方性法规']

    text = " ".join(jieba.cut(text))
    # logger.info(text)
    text_list = text.split(' ')
    query_list = []

    if len(text_list) > 0:
        for one_text in text_list:
            query_list.append({'bool': {'should': [{'match_phrase': {'resultClause': one_text}},
                                                   {'match_phrase': {'title': one_text}}]}})
    if len(sxx_list) > 0:
        query_list.append({"terms": {"isValid.keyword": sxx_list}})
    if len(legal_list) > 0:
        query_list.append({"terms": {"source.keyword": legal_list}})

    query_dict = {
        "query": {"bool": {"must": query_list, }},
        "sort": [
            {'title_weight': {'order': 'desc'}},
            {"isValid_weight": {"order": "asc"}},
            {"legal_type_weight": {"order": "asc"}},
        ],
        "size": size,
    }

    res = search_data_from_es(query_dict)
    return res


if __name__ == '__main__':
    from pprint import pprint

    result = get_law_search_result(text="人民")
    pprint(result)
