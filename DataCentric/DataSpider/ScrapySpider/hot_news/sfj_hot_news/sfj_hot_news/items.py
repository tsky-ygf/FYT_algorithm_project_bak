# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class SfjHotNewsItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # 唯一 主键
    uq_id = scrapy.Field()
    # 标题
    title = scrapy.Field()
    # 内容
    content = scrapy.Field()
    # 带html标签 的内容
    htmlContent = scrapy.Field()
    # 省份
    province = scrapy.Field()
    # 发布时间
    pubDate = scrapy.Field()
    # 网页链接
    url = scrapy.Field()
    # app展示 类别
    category = scrapy.Field()
    # 网站来源
    source = scrapy.Field()
    # 问题类型
    question_type = scrapy.Field()
