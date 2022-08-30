#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/8 14:14 
@Desc    : None
"""
from LegalKnowledge.repository.legal_knowledge_repository import get_news_by_id_list


def test_get_news_by_id_list():
    id_list = [886, 908, 912]
    news = get_news_by_id_list(id_list)

    print(news)
    assert news
    assert len(news) == len(id_list)
