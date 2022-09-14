#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 13:36 
@Desc    : None
"""
import logging
from typing import List, Dict
import re
import pandas as pd
import requests

from LegalKnowledge.constants import RECOMMEND_NEWS_URL, SEARCH_NEWS_URL
from LegalKnowledge.repository import legal_knowledge_repository as repository

_memory = dict()


def get_columns():
    return [
        {
            "column_id": "study_law_daily",
            "column_name": "每日学法"
        },
        {
            "column_id": "hot_news",
            "column_name": "法律热点"
        },
        {
            "column_id": "interpret_the_law_by_case",
            "column_name": "以案释法"
        },
        {
            "column_id": "new_law_express",
            "column_name": "新法速递"
        }
    ]


def _get_column_name(column_id):
    columns = get_columns()
    column_name = next((item.get("column_name") for item in columns if item.get("column_id") == column_id), None)
    if column_name == "法律热点":
        return "新闻热点"
    return column_name


def _has_special_format(column_name):
    return column_name == "每日学法"


def _get_id_list_of_recommended_news(column_name) -> List[int]:
    try:
        resp_json = requests.post(url=RECOMMEND_NEWS_URL).json()
        news = resp_json.get(column_name)

        if _has_special_format(column_name):
            # news is Dict
            return [item[0] for type_news in news.values() for item in type_news]
        # news is List
        return [item[0] for item in news]
    except:
        logging.exception("request error when invoking _get_recommend_news().")
        return []


def _get_news_by_column_name(column_name) -> List[Dict]:
    id_list = _get_id_list_of_recommended_news(column_name)
    return repository.get_news_by_id_list(id_list)
    # mock data
    # result = [
    #     {
    #         "id": 0,
    #         "title": "场馆游泳意外身亡 未尽安保担责六成",
    #         "release_time": "2022-07-13",
    #         "content": "　　□ 本报记者 张驰　　□ 本报见习记者 范瑞恒　　□ 本报通讯员 张晓斌　　炎炎夏日，各类游泳健身场馆正在成为人们“乘风破浪”的好去处，但其中潜在的运动风险也容易引发相关的纠纷。近日，天津市津南区人民法院审理了一起涉及违反安全保障义务责任纠纷的案件。　　2021年夏天，赵某在某游泳馆游泳时身体突发状况不幸离世。现场监控画面显示，自赵某身体进入水中出现溺水现象到工作人员将赵某救出水面前后历时近2分钟，上岸后虽有工作人员对赵某进行心肺复苏，但赵某仍于当日上午被医院宣布死亡，原因是呼吸心跳骤停、溺水。　　赵某家属认为，某游泳馆经营者在馆内未配备专业救生人员及专业医务人员，且在泳客游泳过程中没有救生员进行安全巡视，导致未能及时发现赵某溺水并第一时间进行抢救，错过了赵某的最佳救助时间。遂将某游泳馆诉至法院，要求其对赵某溺水死亡承担赔偿责任。　　法院经审理认为，游泳馆经营者应对游泳参与人员负有更加谨慎的安全保障义务。调查显示，现场工作人员在事发时处在距溺水地点较远一侧的前台位置，未在游泳池边进行不间断巡视，延误了对赵某的抢救，据此应认定该其未能尽到安全保障义务，应当承担相应赔偿责任。　　此外，根据事发当时的监控视频发现，赵某在发生意外情况时并无明显的呼叫或挣扎，结合医院诊断，不排除赵某在事发当时存在导致呼吸心跳骤停的其他因素。由于赵某家属拒绝对赵某进行尸检，导致难以确定赵某死亡的真正原因，据此认定赵某对其损害后果亦存在相应的过错。　　综上，法院最终认定某游泳馆经营者对赵某死亡造成的损失承担60%的赔偿责任。　　法官庭后表示，根据民法典规定，安全保障义务主体承担责任应以其未尽到安全保障义务为前提条件，而安全保障义务又应以合理适当为限度。判断安全保障义务主体是否履行了安全保障义务，可以从法定标准、行业标准、合同标准、善良管理人标准、特别标准五个方面加以把握。　　本案中，由于游泳属于具有危险性的体育项目，某游泳馆经营者作为服务提供者未对游泳参与人履行相应的安全保障义务，对赵某的死亡存在过错，据此作出如上判决。",
    #         "source_url": "http://legalinfo.moj.gov.cn"
    #     }
    # ]
    # return result


def get_news_by_column_id(column_id) -> List[Dict]:
    # fake function
    # df = pd.read_csv("LegalKnowledge/core/recommend_news.csv", encoding="utf-8")
    # return [
    #     {
    #         "id": row["id"],
    #         "title": row["title"],
    #         "release_time": row["release_time"],
    #         "content": row["content"],
    #         "raw_content": row["raw_content"],
    #         "source_url": row["source_url"]
    #     }
    #     for index, row in df.iterrows() if str(row["column_id"]) == str(column_id)
    # ]

    # real function
    column_name = _get_column_name(column_id)
    if column_name is None:
        return []

    news_list = _get_news_by_column_name(column_name)
    for news in news_list:
        _memory[news["id"]] = news
    return news_list


def _get_id_list_after_query(keyword) -> List[int]:
    try:
        resp_json = requests.post(url=SEARCH_NEWS_URL, json={"query": keyword}).json()
        news = []
        for column_name, column_news in resp_json.items():
            news.extend(column_news)
        return [item[0] for item in news]
    except:
        logging.exception("request error when invoking _get_id_list_after_query().")
        return []


def get_news_by_keyword(keyword):
    if keyword is None or str(keyword).strip() == "":
        return []

    id_list = _get_id_list_after_query(keyword)
    return repository.get_news_by_id_list(id_list)


def get_news_by_news_id(news_id):
    # fake function
    # df = pd.read_csv("LegalKnowledge/core/recommend_news.csv", encoding="utf-8")
    # return [
    #     {
    #         "id": row["id"],
    #         "title": row["title"],
    #         "release_time": row["release_time"],
    #         "content": row["content"],
    #         "raw_content": re.sub("href='.+?'", "", row["raw_content"]),
    #         "source_url": row["source_url"]
    #     }
    #     for index, row in df.iterrows() if str(row["id"]) == str(news_id)
    # ]
    # real function
    if news_id in _memory:
        return _memory.get(news_id)
    news = repository.get_news_by_id_list([news_id])
    _memory[news_id] = news
    return news
