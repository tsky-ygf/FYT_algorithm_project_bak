import scrapy


class JilinSpider(scrapy.Spider):
    name = 'jilin'
    allowed_domains = ['www.baidi.com']
    start_urls = ['http://www.baidi.com/']

    def parse(self, response):
        pass
