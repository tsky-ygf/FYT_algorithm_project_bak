#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/8 10:36 
@Desc    : None
"""
from LegalKnowledge.core.legal_knowledge_service import get_news_by_column_id, get_news_by_keyword


def test_get_news_by_column_id():
    column_id = "hot_news"
    news = get_news_by_column_id(column_id)

    print(news)
    assert news
    assert len(news) > 1
    assert isinstance(news[0]["release_time"], str)


def test_get_news_by_keyword():
    keyword = "游泳"
    news = get_news_by_keyword(keyword)

    print(news)
    assert news
    assert len(news) > 1
    assert isinstance(news[0]["release_time"], str)
