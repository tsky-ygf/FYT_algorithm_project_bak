# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FlkcQuestionItem(scrapy.Item):
    # 一级标签
    first_type = scrapy.Field()
    # 二级分类
    second_type = scrapy.Field()
    # 三级分类
    third_type = scrapy.Field()
    # 知识分类
    zhishi_type = scrapy.Field()
    # 链接
    url = scrapy.Field()
    # 唯一 主键
    uq_id = scrapy.Field()
    # 问题
    question = scrapy.Field()
    # 来源
    source = scrapy.Field()
    # 发布时间
    pubDate = scrapy.Field()
    # 回答
    answer = scrapy.Field()
    # html 原文
    htmlContent = scrapy.Field()
