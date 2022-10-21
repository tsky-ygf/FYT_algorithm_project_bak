import scrapy


class ShanghaiSpider(scrapy.Spider):
    name = 'shanghai'
    allowed_domains = ['www.baidi.com']
    start_urls = ['http://www.baidi.com/']

    def parse(self, response):
        pass
