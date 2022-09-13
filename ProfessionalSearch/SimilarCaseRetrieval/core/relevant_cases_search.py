import re

import addressparser
import jieba
import pandas as pd

from ProfessionalSearch.SimilarCaseRetrieval.core.read_case_from_db import search_data_from_es
from Utils.io import read_json_attribute_value


def sort_by_year(res):
    res.insert(res.shape[1], 'year', 0)
    res_new = res
    for index_res, row in res.iterrows():
        if row['event_num'] != '' and row['event_num'] != None:
            year_str = re.findall("(?<=\\（)\\S+(?=\\）)", row['event_num'])
            if year_str:
                res_new.loc[index_res, 'year'] = int(year_str[0])
            else:
                res_new.loc[index_res, 'year'] = 0
    res_new = res_new.sort_values(by="year", ascending=False)
    res_new_drop = res_new.drop(['year'], axis=1, inplace=False)
    return res_new_drop

def filter_region(res, region_list):
    if region_list[0] == "全国":
        return res
    res_drop = res
    for index, row in res_drop.iterrows():
        pro_df = addressparser.transform([row["faYuan_name"]])
        if region_list[0] != pro_df.at[0, "省"]:
            res_drop = res_drop.drop([index])
    return res_drop

def get_case_search_result(text="", type_case_list=None,
                           court_level_list=None,
                           type_document_list=None,
                           region_list=None,
                           page_num=None,
                           page_size=None):
    """
    法条搜索
    :param text: 搜索文本
    :param type_of_case: 案件类型
    :param court_level: 法院层级
    :param type_of_document: 文书类型
    :param region: 地域
    :param size: 搜索结果数量
    :return:
    """
    # if type_case_list is None or type_case_list[0] == '全部':
    #     type_case_list = ['']
    # if court_level_list is None or court_level_list[0] == '全部':
    #     court_level_list = ['']
    # if type_document_list is None or type_document_list[0] == '全部':
    #     type_document_list = ['']
    # if region_list is None or region_list[0] == '全国':
    #     region_list = ['']
    if page_num is None:
        page_num = 1
    if page_size is None:
        page_size = 10

    text = " ".join(jieba.cut(text))
    # logger.info(text)
    text_list = text.split(' ')
    # es查询json
    query_list = []
    bool_value ={}
    if len(text_list) > 0 and text_list[0] != '':
        for one_text in text_list:
            if one_text == '':
                continue
            query_list.append({"match": {"content": {"query": one_text, "boost": 5}}})

    if court_level_list is None or len(court_level_list) == 0 or court_level_list[0] == '' or court_level_list[0] == "全部":
        pass
    elif len(court_level_list) > 0 and '基层' not in court_level_list:
        query_list.append({"match_phrase": {"faYuan_name": {"query": court_level_list[0], "boost": 3}}})

    if type_case_list is None or len(type_case_list) == 0 or type_case_list[0] == '' or type_case_list[0] == "全部":
        pass
    elif len(type_case_list) > 0:
        if type_case_list[0] == "民事":
            type_case = "judgment_minshi_data"
        else:
            type_case = "judgment_xingshi_data"
        query_list.append({"match_phrase": {"table_name": {"query": type_case, "boost": 3}}})

    if type_document_list is None or len(type_document_list) == 0 or type_document_list[0] == '' or type_document_list[0] == "全部":
        pass
    elif len(type_document_list) > 0:
        query_list.append({"match_phrase": {"event_type": {"query": type_document_list[0], "boost": 3}}})

    if region_list and len(region_list) > 0 and region_list[0] != '' and region_list[0] != "全国":
        prev_short_name = read_json_attribute_value("ProfessionalSearch/SimilarCaseRetrieval/api/provin_short_name.json", region_list[0])
        query_list.append({"match_phrase": {"event_num": {"query": prev_short_name, "boost": 5}}})

    bool_value['must'] = query_list
    if court_level_list and len(court_level_list) > 0 and '基层' in court_level_list:  # 基层filter
        bool_value['must_not'] = {"terms": {"faYuan_name.keyword": ['最高', '高级', '中级']}}

    query_dict = {
        "from": page_num,
        "size": page_size,
        "query": {"bool": bool_value},
    }

    res = search_data_from_es(query_dict)
    res_filtered = sort_by_year(res)
    print(res_filtered)
    return res_filtered

if __name__=='__main__':
    get_case_search_result()