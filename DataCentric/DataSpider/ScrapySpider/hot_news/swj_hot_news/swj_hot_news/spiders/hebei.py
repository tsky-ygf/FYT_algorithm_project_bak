import scrapy


class HebeiSpider(scrapy.Spider):
    name = 'hebei'
    allowed_domains = ['www.baidi.com']
    start_urls = ['http://www.baidi.com/']

    def parse(self, response):
        pass
