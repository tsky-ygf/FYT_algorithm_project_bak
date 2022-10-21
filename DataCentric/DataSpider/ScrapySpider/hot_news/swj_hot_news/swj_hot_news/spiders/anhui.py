import scrapy


class AnhuiSpider(scrapy.Spider):
    name = 'anhui'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
