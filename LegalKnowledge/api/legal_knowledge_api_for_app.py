#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 13:17 
@Desc    : 普法常识模块的接口
"""
import traceback
from loguru import logger
import requests
from flask import Flask
from flask import request
from Utils.http_response import response_successful_result, response_failed_result
from LegalKnowledge.core import legal_knowledge_service as service


app = Flask(__name__)


@app.route('/get_columns', methods=["get"])
def get_columns():
    return response_successful_result(service.get_columns())


def _get_short_content(content):
    try:
        short_content = str(content).split("。")[0].split(" ")[-1].split("\u3000")[-1]
        return short_content[:30] + "..."
    except Exception:
        return "..."


def _get_simple_news(news):
    return [
        {
            "id": item["id"],
            "title": str(item["title"]).strip().strip("."),
            "content": _get_short_content(item["content"])
        }
        for item in news
    ]


@app.route('/get_news_by_column_id', methods=["get"])
def get_news_by_column_id():
    column_id = request.args.get("column_id")
    if column_id:
        news = _get_simple_news(service.get_news_by_column_id(column_id))
        return response_successful_result(news, {"total_amount": len(news)})
    return response_failed_result("No parameter: column_id")


@app.route('/get_news_by_keyword', methods=["get"])
def get_news_by_keyword():
    keyword = request.args.get("keyword")
    if keyword:
        news = _get_simple_news(service.get_news_by_keyword(keyword))
        return response_successful_result(news, {"total_amount": len(news)})
    return response_failed_result("No parameter: keyword")


def _get_suffix_content(news_source: str):
    if news_source:
        return "<br>" + "<p>来源 | {}</p>".format(news_source) + "<p>声明 | 本文仅供交流学习，版权归原作者所有，部分文字推送时未能及时与原作者取得联系，若来源标注错误或侵犯到您的权益，烦请告知删除。</p>"
    return "<br>" + "<p>声明 | 本文仅供交流学习，版权归原作者所有，部分文字推送时未能及时与原作者取得联系，若来源标注错误或侵犯到您的权益，烦请告知删除。</p>"


def _get_detailed_news(news):
    return [
        {
            "id": item["id"],
            "title": str(item["title"]).strip().strip("."),
            "release_time": item["release_time"],
            "content": _get_short_content(item["content"]),
            "raw_content": str(item["raw_content"]) + _get_suffix_content(item["source_char"]),
            "source_url": item["source_url"]
        }
        for item in news
    ]


@app.route('/get_news_by_news_id', methods=["get"])
def get_news_by_news_id():
    news_id = request.args.get("news_id")
    if news_id:
        news = _get_detailed_news(service.get_news_by_news_id(news_id))
        return response_successful_result(news[0] if news else dict())
    return response_failed_result("No parameter: news_id")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8122, debug=False)
