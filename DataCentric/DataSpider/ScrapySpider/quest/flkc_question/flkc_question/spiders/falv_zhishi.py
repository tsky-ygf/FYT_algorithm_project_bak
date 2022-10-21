import re

import scrapy
from lxml import etree

class FalvZhishiSpider(scrapy.Spider):
    name = 'falv_zhishi'
    start_urls = ['https://www.lawtime.cn/zhishi/']

    def parse(self, response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        # html.xpath("//li[@class='menu-classify-list-item']")
        # box_lit = html.xpath("//div[contains(@class,'menu-slide-box')]/ul/li[@class='menu-slide-list-item']//text()")
        # box_list = html.xpath("//div[contains(@class,'menu-slide-box')]")[:-1]
        li_list = html.xpath("//li[@class='menu-classify-list-item']")[:-1]
        for li in li_list:
            zhishi_type = re.sub("\s","","".join(li.xpath("./div[1]//text()")))
            a_list = li.xpath("./div[contains(@class,'menu-slide-box')]/ul/li[@class='menu-slide-list-item']/a")
            for a in a_list:
                href = a.xpath("./@href")[0]
                first_type = a.xpath("./text()")[0]
                url = href + "list.html"
                print(zhishi_type,first_type,url)
                # yield scrapy.Request(url=url, dont_filter=True, callback=self)
                break
            # break

