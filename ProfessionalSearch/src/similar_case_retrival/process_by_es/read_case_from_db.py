import pandas as pd
from elasticsearch import Elasticsearch


def search_data_from_es(
    query_body, _index_name="case_index_v2", _es_hosts="127.0.0.1:9200"
):
    # 查询数据
    es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True, hosts=_es_hosts)
    res = es.search(index=_index_name, body=query_body)
    print(res)
    res_list = [hit["_source"] for hit in res["hits"]["hits"]]
    df = pd.DataFrame(res_list)
    df.fillna("", inplace=True)
    return df, res["hits"]["total"]["value"]


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
        "size": 100,
        "query": {
            "bool": {
                "must": [
                    # {"match_phrase": {"content": "石家庄市"}},
                    # {"match_phrase": {"faYuan_name": "中级"}},
                    # {"match_phrase": {"jfType": "合同纠纷"}},
                    # {"match_phrase": {"event_type": "裁定书"}},
                    {
                        "match": {
                            "faYuan_name": {
                                "query": "最高",
                                # "operator": "and"
                                "boost": 3,
                            }
                        }
                    },
                    # {"match_phrase": {"jfType": {"query": "合同纠纷",
                    #                       "boost": 3
                    #                           # "operator": "and"
                    #                           }}},
                    {
                        "match_phrase": {
                            "event_type": {
                                "query": "裁定",
                                "boost": 3
                                # "operator": "and"
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "table_name": {"query": "judgment_minshi_data", "boost": 3}
                        }
                    },
                    # {"match": {"event_num": {"query": "青",
                    #                        "boost": 5,
                    #                        # "operator": "and"
                    #                        }}},
                    {
                        "match": {
                            "content": {
                                "query": "广东省",
                                "boost": 5,
                                # "operator": "and"
                            }
                        }
                    },
                    # {"match": {"content": {"query": "契约",
                    #                        "boost": 5,
                    #                        # "operator": "and"
                    #                        }}},
                ],
            }
        },
    }

    query_dict = {
        "from": 1,
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    {"match_phrase": {"content": {"query": "买卖", "boost": 3}}},
                    {"match_phrase": {"faYuan_name": {"query": "最高", "boost": 3}}},
                    {
                        "span_containing": {
                            "big": {
                                "span_near": {
                                    "clauses": [
                                        {"span_term": {"content": "裁"}},
                                        {"span_term": {"content": "定"}},
                                    ],
                                    "slop": 30,
                                    "in_order": True,
                                }
                            },
                            "little": {
                                "span_first": {
                                    "match": {
                                        "span_term": {"content": "裁"},
                                        "span_term": {"content": "定"},
                                    },
                                    "end": 30,
                                }
                            },
                        }
                    },
                    {"match_phrase": {"db_name": {"query": "judgments_mingshi_data"}}},
                    # {"match_phrase": {"province": {"query": "安徽省"}}},
                ]
            }
        },
    }

    # query_dict = {
    #     "from": 1,
    #     "size": 10,
    #     "query": {
    #         "match_all": {}
    #         },
    #      "script_fields": {
    #          "content": {
    #              "source":"doc['content'].value.substring(0,'）').contain('广东')"
    #          }
    #      }
    #     }
    # type_case_list = ["调解"]
    # type_case_sub = list(type_case_list[0])
    # query_dict = {
    #     "from": 1,
    #     "size": 10,
    #     "query": {
    #         "bool": {
    #             "must": [
    #                 {
    #                     "span_first": {
    #                         {
    #                             "match":{
    #                                 "span_term": {"content": "通知书"}
    #                             },
    #                             "end": 50
    #                          }
    #                     }
    #                 },
    #                 # {"match_phrase": {"event_type": {"query": "执行裁定书", "boost": 3}}}, a83de20f4b9ec9f957f7307941fb5c78
    #                 {"match_phrase": {"table_name": {"query": "judgment_zhixing_data", "boost": 3}}},
    #             ]
    #         }
    #     },
    # }
    # query_dict = {"query":
    # {
    #     "span_containing": {
    #         "big": {
    #             "span_near": {
    #                 "clauses":[
    #                     {"span_term": {"content": "裁"}},
    #                     {"span_term": {"content": "定"}}
    #                 ],
    #                 "slop": 30,
    #                 "in_order" : True
    #                 }
    #             },
    #         "little": {
    #             "span_first": {
    #                 "match": {
    #                     "span_term": {"content": "裁"},
    #                     "span_term": {"content": "定"},
    #                 },
    #                 "end": 30
    #             }
    #         }
    #       }
    #     }
    # }
    query_dict = {
        "from": 1,
        "size": 10,
        "query": {
            "bool": {
                "must": [
                    {"match": {"content": {"query": "买卖", "boost": 5}}},
                    {"match_phrase": {"faYuan_name": {"query": "高级", "boost": 3}}},
                    {
                        "match_phrase": {
                            "table_name": {
                                "query": "judgment_minshi_data_cc",
                                "boost": 3,
                            }
                        }
                    },
                    # {'match_phrase': {'event_type': {'query': '判决', 'boost': 3}}},
                    {"match_phrase": {"province": {"query": "浙江", "boost": 15}}},
                ]
            }
        },
    }

    query_dict = {
        "explain": True,
        "query": {
            "bool": {
                "should": [
                    {"match": {"jfType": {"query": "婚姻家庭", "boost": 10}}},
                    {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "chaming": "利 息   存 在   签 订   借 贷   借 款   支 付   约 定 果   请 具体 描述    借贷 纠纷   还本付息 双方 自愿 自愿 签订 签订 借款 协议 欠条 收据 通过 转账 现金 方式 支付 支付 借款 钱款 用于 用途 借款人 到期 足额 偿还 支付 本金 利息 利息 存在 存在 人为 担保人 约定 约定 借款 利息 借款 期限 利息 年利率 借贷 过程 存在 行为 导致 后果 具体 描述 借贷 纠纷 还本付息"
                                    }
                                },
                                {
                                    "match": {
                                        "benyuan_renwei": "利 息   存 在   签 订   借 贷   借 款   支 付   约 定 果  请 具体 描述    借贷 纠纷   还本付息 双方 自愿 自愿 签订 签订 借款 协议 欠条 收据 通过 转账 现金 方式 支付 支付 借款 钱款 用于 用途 借款人 到期 足额 偿还 支付 本金 利息 利息 存在 存在 人为 担保人 约定 约定 借款 利息 借款 期限 利息 年利率 借贷 过程 存在 行为 导致 后果 具体 描述 借贷 纠纷 还本付息"
                                    }
                                },
                                {
                                    "match": {
                                        "sucheng_sentences": "利 息   存 在   签 订   借 贷   借 款   支 付   约 定 果  请 具体 描述    借贷 纠纷   还本付息 双方 自愿 自愿 签订 签订 借款 协议 欠条 收据 通过 转账 现金 方式 支付 支付 借款 钱款 用于 用途 借款人 到期 足额 偿还 支付 本金 利息 利息 存在 存在 人为 担保人 约定 约定 借款 利息 借款 期限 利息 年利率 借贷 过程 存在 行为 导致 后果 具体 描述 借贷 纠纷 还本付息"
                                    }
                                },
                            ]
                        }
                    },
                ]
            }
        }
    }

    # query_dict = {
    #     "query": {
    #         "bool": {
    #             "should": [
    #                 {"match": {"jfType": {"query": "婚姻家庭", "boost": 10}}},
    #                 {
    #                     "bool": {
    #                         "should": [
    #                             {
    #                                 "match": {
    #                                     "chaming": "还本付息"
    #                                 }
    #                             },
    #                             {
    #                                 "match": {
    #                                     "benyuan_renwei": "还本付息"
    #                                 }
    #                             },
    #                             {
    #                                 "match": {
    #                                     "sucheng_sentences": "还本付息"
    #                                 }
    #                             },
    #                         ]
    #                     }
    #                 },
    #             ]
    #         }
    #     }
    # }

    res_df, total_num = search_data_from_es(query_dict)
    # res_df = pd.DataFrame(columns=[''])
    # res = _construct_result_format(res_df)
    print(res_df)
    for index, row in res_df.iterrows():
        print(row.to_dict())
