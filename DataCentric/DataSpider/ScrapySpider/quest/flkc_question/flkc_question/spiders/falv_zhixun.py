import scrapy


class FalvZhixunSpider(scrapy.Spider):
    name = 'falv_zhixun'
    allowed_domains = ['www.baidu.com']
    start_urls = ['http://www.baidu.com/']

    def parse(self, response):
        pass
