import scrapy


class HenanSpider(scrapy.Spider):
    name = 'henan'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
