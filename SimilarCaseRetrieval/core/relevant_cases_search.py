import jieba

from SimilarCaseRetrieval.core.read_case_from_db import search_data_from_es


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
    if type_case_list is None:
        type_case_list = ['']
    if court_level_list is None:
        court_level_list = ['']
    if type_document_list is None:
        type_document_list = ['']
    if region_list is None:
        region_list = ['']
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
    if len(text_list) > 0:
        for one_text in text_list:
            query_list.append({'bool': {'should': [{'match_phrase': {'content': one_text}}]}})

    if court_level_list is None or len(court_level_list) == 0:
        pass
    elif len(court_level_list) > 0 and '基层' not in court_level_list:
        query_list.append({'bool': {'should': [{'match_phrase': {'faYuan_name': court_level_list[0]}}]}})

    if type_case_list is None or len(type_case_list) == 0:
        pass
    elif len(type_case_list) > 0:
        query_list.append({'bool': {'should': [{'match_phrase': {'jfType': type_case_list[0]}}]}})

    if type_document_list is None or len(type_document_list) == 0:
        pass
    elif len(type_document_list) > 0:
        query_list.append({'bool': {'should': [{'match_phrase': {'event_type': type_document_list[0]}}]}})

    if region_list is None or len(region_list) == 0:
        pass
    elif len(region_list) > 0:
        query_list.append({'bool': {'should': [{'match_phrase': {'content': region_list[0]}}]}})
    bool_value['should'] = query_list

    if region_list is not None and len(court_level_list) > 0 and '基层' in court_level_list:  # 基层filter
        bool_value['must_not'] = {"terms": {"faYuan_name.keyword": ['最高', '高级', '中级']}}

    query_dict = {
        "from": page_num,
        "size": page_size,
        "query": {"bool": bool_value},

    }

    res = search_data_from_es(query_dict)
    print(res)
    return res
    pass

if __name__=='__main__':
    get_case_search_result()