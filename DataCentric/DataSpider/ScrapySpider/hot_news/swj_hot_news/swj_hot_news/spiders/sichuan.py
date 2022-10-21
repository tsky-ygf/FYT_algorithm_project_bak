import scrapy


class SichuanSpider(scrapy.Spider):
    name = 'sichuan'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
