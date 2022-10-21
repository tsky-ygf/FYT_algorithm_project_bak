# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class QuestItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # 问题
    question = scrapy.Field()
    # 回答
    answer = scrapy.Field()
    # 发布时间
    pubData = scrapy.Field()
    # 问题类型 标签
    question_sign = scrapy.Field()
    # 热点问题标签
    hot_question_sign = scrapy.Field()
    # 问题类型
    question_type = scrapy.Field()
    # 唯一主键
    uq_id = scrapy.Field()
    # 来源
    source = scrapy.Field()
    model_type = scrapy.Field()
    # url 链接
    url = scrapy.Field()
