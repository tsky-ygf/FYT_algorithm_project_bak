import scrapy


class BeijingSpider(scrapy.Spider):
    name = 'beijing'
    allowed_domains = ['www.baidi.com']
    start_urls = ['http://www.baidi.com/']

    def parse(self, response):
        pass
