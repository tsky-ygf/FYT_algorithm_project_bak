import scrapy


class JiangsuSpider(scrapy.Spider):
    name = 'jiangsu'
    allowed_domains = ['www.baidi.com']
    start_urls = ['http://www.baidi.com/']

    def parse(self, response):
        pass
