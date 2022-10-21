import scrapy


class ShandongSpider(scrapy.Spider):
    name = 'shandong'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
