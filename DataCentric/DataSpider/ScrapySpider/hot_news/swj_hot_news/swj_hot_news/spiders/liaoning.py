import scrapy


class LiaoningSpider(scrapy.Spider):
    name = 'liaoning'
    allowed_domains = ['www.baidi.com']
    start_urls = ['http://www.baidi.com/']

    def parse(self, response):
        pass
