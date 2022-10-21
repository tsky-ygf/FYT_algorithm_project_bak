import scrapy


class GuangdongSpider(scrapy.Spider):
    name = 'guangdong'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
