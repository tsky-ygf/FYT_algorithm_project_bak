import scrapy


class JiangxiSpider(scrapy.Spider):
    name = 'jiangxi'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
