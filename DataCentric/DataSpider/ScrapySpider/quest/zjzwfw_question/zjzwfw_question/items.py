# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ZjzwfwQuestionItem(scrapy.Item):
    # 唯一 主键
    uq_id = scrapy.Field()
    # 问题 id
    storeid = scrapy.Field()
    # 问题
    question = scrapy.Field()
    # 问题标题
    quest_title = scrapy.Field()
    # 回答
    answer = scrapy.Field()
    # 回答单位
    ans_person = scrapy.Field()
    # 目标
    goal = scrapy.Field()
    # 来源
    source = scrapy.Field()
    # 问题来源
    quest_source = scrapy.Field()
    # 发布时间
    pubDate = scrapy.Field()
    # 状态
    state = scrapy.Field()
    # 城市
    city = scrapy.Field()
    # 省份
    province = scrapy.Field()

