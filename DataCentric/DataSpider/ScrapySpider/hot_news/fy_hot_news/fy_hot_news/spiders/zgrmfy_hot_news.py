import copy
import hashlib

import scrapy
from lxml import etree
import redis
from fy_hot_news.items import FyHotNewsItem


def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class ZgrmfyHotNewsSpider(scrapy.Spider):
    name = 'fy_hot_news'
    redis_key = "fy_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    start_urls = ['https://www.court.gov.cn/zixun.html']

    def parse(self, response, **kwargs):
        item = FyHotNewsItem()
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        a_list = html.xpath("//ul[@id='yw1']/li/a")[:3]
        for a in a_list:
            href = a.xpath("./@href")[0]
            question_type = a.xpath("./text()")[0]
            item['question_type'] = question_type
            url = "https://www.court.gov.cn" + href
            yield scrapy.Request(url=url, callback=self.zgfy_get_page, dont_filter=True, meta={'item':copy.deepcopy(item)})

    def zgfy_get_page(self, response, **kwargs):
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        sum_total = html.xpath("//span[@class='num']/text()")[0]
        max_page = int(int(sum_total)/20) +1
        url = response.url
        for page in range(1,max_page + 1):
            new_url = url + f"?page={page}"
            yield scrapy.Request(url=new_url, dont_filter=True, meta={'item':copy.deepcopy(item)}, callback=self.zgfy_get_detail_url)
            print(new_url)

    def zgfy_get_detail_url(self, response, **kwargs):
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        li_list = html.xpath("//div[@class='sec_list']/ul/li")
        for li in li_list:
            title = li.xpath("./a/@title")[0]
            href = li.xpath("./a/@href")[0]
            pubDate = li.xpath("./i/text()")[0]
            url = "https://www.court.gov.cn" + href
            item['title'] = title
            item['pubDate'] = pubDate
            item['url'] = url
            item['uq_id'] = encode_md5(url)
            res = self.redis_conn.sadd(self.redis_key, url)
            if res == 1:
                yield scrapy.Request(url=url, callback=self.zgfy_get_detail, dont_filter=True, meta={'item':copy.deepcopy(item)})

    def zgfy_get_detail(self, response, **kwargs):
        item = response.meta['item']
        text = response.body.decode('utf-8')
        # print(text)
        html = etree.HTML(text)
        content_div = html.xpath('//*[@id="container"]/div/div[contains(@class,"txt")]/div[1]')[0]
        htmlContent = etree.tostring(content_div, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        content = ''.join(content_div.xpath('.//text()'))
        print("*"*80)
        item['category'] = "法院"
        item['source'] = "中华人民共和国最高人民法院"
        item['htmlContent'] = htmlContent
        item['content'] = content
        yield item
        # htmlContent = etree.tostring(content_div, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        # content = "".join(content_div.xpath(".//text()"))
        # print(content)
        # https://www.court.gov.cn/zixun-gengduo-23.html?page=1

