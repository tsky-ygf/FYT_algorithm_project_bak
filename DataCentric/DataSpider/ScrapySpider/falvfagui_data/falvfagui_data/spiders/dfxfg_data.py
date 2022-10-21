import scrapy


class DfxfgDataSpider(scrapy.Spider):
    name = 'dfxfg_data'
    allowed_domains = ['www.baidu.com']
    start_urls = ['http://www.baidu.com/']

    def parse(self, response):
        pass
