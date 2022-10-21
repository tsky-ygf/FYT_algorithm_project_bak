import scrapy


class HainanSpider(scrapy.Spider):
    name = 'hainan'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
