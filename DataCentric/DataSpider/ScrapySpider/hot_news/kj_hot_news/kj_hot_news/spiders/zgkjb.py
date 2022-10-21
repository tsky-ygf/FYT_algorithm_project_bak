import copy
import hashlib
import re
from lxml import etree
import scrapy
import redis
from kj_hot_news.items import KjHotNewsItem

def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class ZgkjbSpider(scrapy.Spider):
    name = 'zgkjb'
    start_urls = [
        'https://www.most.gov.cn/dfkj/dfkjyw/dfzxdt/',  # 地方动态
        'https://www.most.gov.cn/gnwkjdt/',  # 科技动态
    ]
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    redis_key = "jt_hot_news"

    def parse(self, response):
        resp_url = response.url
        yield scrapy.Request(url=resp_url, dont_filter=True, callback=self.get_detail_url)
        for i in range(1,26):
            url = resp_url + f"index_{i}.html"
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_url)

    def get_detail_url(self, response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        item = KjHotNewsItem()
        resp_url = response.url
        if "dfzxdt" in resp_url:
            item["question_type"] = "地方科技"
            href_list = html.xpath("//div[@class='content-list']/ul/li/a/@href")
            for href in href_list:
                url = href.replace("../../","https://www.most.gov.cn/dfkj/")
                if self.redis_conn.sadd(self.redis_key, url):
                    yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail, meta={"item":copy.deepcopy(item)})
                # break
        elif "gnwkjdt" in resp_url:
            item["question_type"] = "科技动态"
            href_list = html.xpath("//div[@class='list-main']/ul/li/a/@href")
            for href in href_list:
                url = href.replace("./","https://www.most.gov.cn/gnwkjdt/")
                if self.redis_conn.sadd(self.redis_key, url):
                    yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail, meta={"item": copy.deepcopy(item)})
                # break
        else:
            return
    def get_detail(self, response):
        item = response.meta['item']
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        resp_url = response.url
        front_url = re.findall("(.*\/)", resp_url)[0]
        div = html.xpath("//div[@class='TRS_Editor']")[0]
        title = html.xpath("//meta[@name='ArticleTitle']/@content")[0]
        pubDate = html.xpath("//meta[@name='PubDate']/@content")[0][:10]
        htmlContent = etree.tostring(div, encoding='utf-8', method='html', pretty_print=True).decode('utf-8')
        htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
        content = "".join(div.xpath(".//text()"))
        item["url"] = resp_url
        item["uq_id"] = encode_md5(resp_url)
        item["title"] = title
        item["content"] = content
        item["htmlContent"] = htmlContent
        item["pubDate"] = pubDate
        item["source"] = "中华人民共和国科学技术部"
        item["category"] = "科技专栏"
        yield item
