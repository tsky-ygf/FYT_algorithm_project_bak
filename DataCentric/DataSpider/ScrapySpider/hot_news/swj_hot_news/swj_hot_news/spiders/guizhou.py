import scrapy


class GuizhouSpider(scrapy.Spider):
    name = 'guizhou'
    allowed_domains = ['www.shanxi.com']
    start_urls = ['http://www.shanxi.com/']

    def parse(self, response):
        pass
