import hashlib

import scrapy
import re
from lxml import etree
from swj_hot_news.items import SwjHotNewsItem


class TianjinSpider(scrapy.Spider):
    name = 'tianjin'
    start_urls = ['https://tianjin.chinatax.gov.cn/u_zlmViewMx.action']

    def start_requests(self):
        data = {
            "lmdm": "020001",
            "fjdm": "11200000000",
            "page": "1",
            "d": ""
        }
        for url in self.start_urls:
            yield scrapy.FormRequest(url=url,formdata=data,callback=self.parse)

    def parse(self, response):
        text = response.body.decode('utf-8')
        # print(text)
        pageCount = re.findall("pageCount: (.*?),", text)[0]
        url = 'https://tianjin.chinatax.gov.cn/u_zlmViewMx.action'
        for page in range(1,int(pageCount)+1):  # int(pageCount)+1
            data = {
                "lmdm": "020001",
                "fjdm": "11200000000",
                "page": f"{page}",
                "d": ""
            }
            yield scrapy.FormRequest(url=url,formdata=data,callback=self.get_url_list)

    def get_url_list(self,response):
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        href_list = re.findall("<a data-audio.*? href=\"(.*?)\"",text)
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36",
            "sec-ch-ua": "\"Google Chrome\";v=\"105\", \"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"105\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\""
        }
        for href in href_list:
            url = "https://tianjin.chinatax.gov.cn/" + href
            yield scrapy.Request(url=url,headers=headers,callback=self.get_detail)
            # break

    def get_detail(self,response):
        item = SwjHotNewsItem()
        text = response.body.decode('utf-8')
        print(text)
        url = response.url
        html = etree.HTML(text)
        title = html.xpath("//div[@id='filetitle']/text()")
        if title:
            title = re.sub("\s","",''.join(title))
            item['title'] = title
        pubDate = re.findall("发布时间：(.*?) ",text)[0]
        content = html.xpath("//td[@id='conntentNR']//text()")
        content = ''.join(content)
        item['content'] = content
        item['htmlContent'] = text
        item['url'] = url
        item['pubDate'] = pubDate
        item['province'] = '天津'
        item['uq_id'] = self.encode_md5(url)
        yield item


    def encode_md5(self,value):
        if value:
            hl = hashlib.md5()
            hl.update(value.encode(encoding='utf-8'))
            return hl.hexdigest()
        return ''