import copy
import hashlib
import re

import scrapy
from lxml import etree
import redis
from sfj_hot_news.items import SfjHotNewsItem


def get_quest_type(url):
    if "yfzldfyfzl" in url:
        return "地方依法治理"
    elif "yfzlhyyfzl" in url:
        return "行业依法治理"
    elif "yfzljcyfzl" in url:
        return "基层依法治理"
    elif "yfzlfzcj" in url:
        return "法治创建"
    else:
        return ''

def encode_md5(str):
    if str:
        hl = hashlib.md5()
        hl.update(str.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class ZhpfSpider(scrapy.Spider):
    # 智慧普法平台
    name = 'zhpf'
    redis_key = "sfj_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    start_urls = [
        'http://legalinfo.moj.gov.cn/pub/sfbzhfx/zhfxyfzl/yfzldfyfzl/',  # 地方依法治理
        'http://legalinfo.moj.gov.cn/pub/sfbzhfx/zhfxyfzl/yfzlhyyfzl/',  # 行业依法治理
        'http://legalinfo.moj.gov.cn/pub/sfbzhfx/zhfxyfzl/yfzljcyfzl/',  # 基层依法治理
        'http://legalinfo.moj.gov.cn/pub/sfbzhfx/zhfxyfzl/yfzlfzcj/',  # 法治创建
    ]

    def parse(self, response, **kwargs):
        # http://legalinfo.moj.gov.cn/pub/sfbzhfx/zhfxyfzl/yfzldfyfzl/index_2.html
        url = response.url
        yield scrapy.Request(url=url, dont_filter=True, callback=self.get_news_url)
        for page in range(1, 11):
            new_url = url + f"index_{page}.html"
            yield scrapy.Request(url=new_url, dont_filter=True,callback=self.get_news_url)
            # break

    def get_news_url(self, response, **kwargs):
        item = SfjHotNewsItem()
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        li_list = html.xpath("//ul[@class='rightSide_list']/li")
        for li in li_list:
            url = li.xpath("./a/@href")[0]
            title = li.xpath("normalize-space(./a/text())")
            pubDate = li.xpath("normalize-space(./span/text())")
            item['pubDate'] = re.sub("\[|\]| ", "", pubDate)
            item['url'] = url
            item['uq_id'] = encode_md5(url)
            item['title'] = title
            item['source'] = "智慧普法平台"
            item["question_type"] = get_quest_type(url)
            res = self.redis_conn.sadd(self.redis_key, url)
            if res == 1:
                yield scrapy.Request(url=url, dont_filter=True,callback=self.get_detail, meta={'item': copy.deepcopy(item)})
            # break

    def get_detail(self, response, **kwargs):
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        content = html.xpath("//div[@id='zhengwen']")[0]
        htmlContent = html.xpath("//div[@id='zhengwen']")[0]
        htmlContent = etree.tostring(htmlContent, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        if content:
            item['content'] = ''.join(content.xpath(".//text()"))
            item['htmlContent'] = htmlContent
            yield item
