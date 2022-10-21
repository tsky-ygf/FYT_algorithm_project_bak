import scrapy


class FujianSpider(scrapy.Spider):
    name = 'fujian'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
