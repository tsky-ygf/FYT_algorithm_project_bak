import scrapy


class HunanSpider(scrapy.Spider):
    name = 'hunan'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
