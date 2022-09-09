#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/8 13:41
# @Author  : Adolf
# @Site    :
# @File    : relevant_laws_search.py
# @Software: PyCharm
import jieba
from ProfessionalSearch.RelevantLaws.LegalLibrary.read_legal_from_db import search_data_from_es

import addressparser


def filter_scope_of_use(res, scope_of_use):
    if scope_of_use[0] == "全国":
        return res
    res_drop = res
    for index, row in res_drop.iterrows():
        pro_df = addressparser.transform([row["locality"]])
        if scope_of_use[0] != pro_df.at[0, "省"]:
            res_drop = res_drop.drop([index])
    return res_drop


def get_law_search_result(
    text="",
    sxx_list=None,
    legal_list=None,
    scope_of_use=None,
    page_number=None,
    page_size=None,
):
    """
    法条搜索
    :param text: 搜索文本
    :param sxx_list: 时效性列表
    :param legal_list: 法条种类列表
    :param size: 搜索结果数量
    :return:
    """
    if sxx_list is None:
        sxx_list = ["有效", "已修改", "尚未生效", "已废止"]
    if legal_list is None:
        legal_list = ["宪法", "法律", "行政法规", "监察法规", "司法解释", "地方性法规"]
    if page_number is None:
        page_number = 1
    if page_size is None:
        page_size = 10
    if scope_of_use is None:
        scope_of_use = ["全国"]
    text = " ".join(jieba.cut(text))
    # logger.info(text)
    text_list = text.split(" ")
    query_list = []

    if text_list[0] != "" and len(text_list) > 0:
        for one_text in text_list:
            query_list.append(
                {
                    "bool": {
                        "should": [
                            {"match_phrase": {"resultClause": one_text}},
                            {"match_phrase": {"title": one_text}},
                        ]
                    }
                }
            )
    if scope_of_use is not None and len(scope_of_use) > 0 and scope_of_use[0] == "全国":
        pass

    if len(sxx_list) > 0 and "全部" not in sxx_list:
        query_list.append({"terms": {"isValid.keyword": sxx_list}})
    else:
        pass
    if len(legal_list) > 0 and "全部" not in legal_list:
        query_list.append({"terms": {"source.keyword": legal_list}})
    else:
        pass

    query_dict = {
        "from": page_number,
        "size": page_size,
        "query": {"bool": {"must": query_list,}},
        "sort": [
            {"title_weight": {"order": "desc"}},
            {"isValid_weight": {"order": "asc"}},
            {"legal_type_weight": {"order": "asc"}},
        ],
    }

    res = search_data_from_es(query_dict)
    if scope_of_use is not None and len(scope_of_use) > 0:
        res_filtered_scope = filter_scope_of_use(res, scope_of_use)
    print(res_filtered_scope)
    return res_filtered_scope


if __name__ == "__main__":
    from pprint import pprint

    result = get_law_search_result(text="人民")
    pprint(result)
