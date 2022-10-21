# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BankHotNewsItem(scrapy.Item):
    # 网页链接
    url = scrapy.Field()
    # 唯一 主键
    uq_id = scrapy.Field()
    # 标题
    title = scrapy.Field()
    # 内容
    content = scrapy.Field()
    htmlContent = scrapy.Field()
    # 省份
    province = scrapy.Field()
    # 发布时间
    pubDate = scrapy.Field()
    source = scrapy.Field()
    category = scrapy.Field()
