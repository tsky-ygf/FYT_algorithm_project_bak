import scrapy


class HeilongjiangSpider(scrapy.Spider):
    name = 'heilongjiang'
    allowed_domains = ['www.baidi.com']
    start_urls = ['http://www.baidi.com/']

    def parse(self, response):
        pass
