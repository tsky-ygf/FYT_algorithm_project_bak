import pandas as pd
from elasticsearch import Elasticsearch

from Utils import print_run_time

# @print_run_time
def search_data_from_es(
    query_body, _index_name="case_index", _es_hosts="127.0.0.1:9200"
):
    # 查询数据
    es = Elasticsearch(hosts=_es_hosts)
    res = es.search(index=_index_name, body=query_body)
    res_list = [hit["_source"] for hit in res["hits"]["hits"]]
    df = pd.DataFrame(res_list)
    df.fillna("", inplace=True)
    return df


def _construct_result_format(search_result):
    result = []
    for index, row in search_result.iterrows():

        result.append(
            {
                "doc_id": row["uq_id"],
                "court": row["faYuan_name"],
                "case_number": row["event_num"],
                "content": row["content"],
            }
        )
    return result


if __name__ == "__main__":
    query_dict = {
        "query": {
            "bool": {
                "must": [
                    {"bool": {"should": [{"match_phrase": {"content": "财产"}}]}},
                    {"bool": {"should": [{"match_phrase": {"content": "离婚"}}]}},
                    {"terms": {"faYuan_name.keyword": ["最高"]}},
                    {"terms": {"jfType.keyword": ["婚姻家庭"]}},
                    {"terms": {"event_type.keyword": ["判决", "裁定", "调解"]}},
                    {"terms": {"content.keyword": ["江苏"]}},
                ],
                "must_not": {"terms": {"faYuan_name.keyword": ["最高", "高级", "中级"]}},
            }
        },
        "size": 10,
    }

    query_dict = {
        "from": 1,
        "size": 1,
        "query": {
            "bool": {
                "should": [
                    # {'bool': {'should': [{'match_phrase': {'content': "家庭"}}]}}, # query
                    # {'bool': {'should': [{'match_phrase': {'content': "离婚"}}]}}, # query
                    {"bool": {"should": [{"match_phrase": {"content": "江苏"}}]}},  # 地域
                    # {'bool': {'should': [{'match_phrase': {'faYuan_name': '中级'}}]}},  # 非基层法院过滤
                    {
                        "bool": {"should": [{"match_phrase": {"faYuan_name": "高级"}}]}
                    },  # 非基层法院过滤
                    {
                        "bool": {"should": [{"match_phrase": {"event_type": "裁定书"}}]}
                    },  # 非基层法院过滤
                    {
                        "bool": {"should": [{"match_phrase": {"jfType": "返还原物纠纷"}}]}
                    },  # 非基层法院过滤
                    # {'terms': {'faYuan_name.keyword': ['最高']}},
                    # {'terms': {'jfType.keyword': ['离婚纠纷']}},
                    # {'terms': {'event_type.keyword': ['判决书']}},
                    # {'terms': {'content.keyword': ['台湾']}}
                ],
                # 'must_not':
                # {"terms": {"faYuan_name.keyword": ['最高', '高级', '中级']}} # 基层法院过滤
            }
        },
    }
    query_dict = {
        "from": 1,
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    # {"match_phrase": {"content": "石家庄市"}},
                    # {"match_phrase": {"faYuan_name": "中级"}},
                    # {"match_phrase": {"jfType": "合同纠纷"}},
                    # {"match_phrase": {"event_type": "裁定书"}},
                    {"match": {"faYuan_name": {"query": "中级",
                                              # "operator": "and"
                                               "boost": 3
                                              }}},
                    {"match_phrase": {"jfType": {"query": "合同纠纷",
                                          "boost": 3
                                              # "operator": "and"
                                              }}},

                    {"match": {"event_type": {"query": "裁定书",
                                              "boost": 3
                                           # "operator": "and"
                                           }}},
                    {"match": {"event_num": {"query": "青",
                                           "boost": 5,
                                           # "operator": "and"
                                           }}},
                    {"match": {"content": {"query": "买卖",
                                           "boost": 5,
                                            # "operator": "and"
                                      }}},
                    {"match": {"content": {"query": "契约",
                                           "boost": 5,
                                           # "operator": "and"
                                           }}},
                ],
            }
        }
    }

    res_df = search_data_from_es(query_dict)
    # res_df = pd.DataFrame(columns=[''])
    # res = _construct_result_format(res_df)
    print(res_df)
    for index, row in res_df.iterrows():
        print(row.to_dict())
