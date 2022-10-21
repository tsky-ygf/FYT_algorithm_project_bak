import scrapy


class HubeiSpider(scrapy.Spider):
    name = 'hubei'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
