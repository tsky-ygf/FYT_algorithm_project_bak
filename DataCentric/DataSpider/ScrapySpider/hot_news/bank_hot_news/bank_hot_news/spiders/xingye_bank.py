import copy
import hashlib
import re

import redis
import scrapy
from lxml import etree
from bank_hot_news.items import BankHotNewsItem

def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class XingyeBankSpider(scrapy.Spider):
    name = 'xingye_bank'
    allowed_domains = ['www.cib.com']
    start_urls = ['https://www.cib.com.cn/cn/aboutCIB/about/news/']
    redis_key = "banking_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)

    def parse(self, response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        item = BankHotNewsItem()
        # href = html.xpath("//li[@class='clearfix']/a/@href")[0]
        # title = html.xpath("//li[@class='clearfix']/a/text()")[0]
        li_list = html.xpath("//li[@class='clearfix']")
        for li in li_list:
            href = li.xpath("./a/@href")[0]
            url = "https://www.cib.com.cn" + href
            title = li.xpath("./a/text()")[0]
            pubDate = li.xpath("./span[@class='time']/text()")[0]
            item['url'] = url
            item['uq_id'] = encode_md5(url)
            item["title"] = "【兴业银行】" + title
            item["pubDate"] = pubDate
            item["source"] = "兴业银行"
            item["category"] = "金融银行"
            # print(item)
            if self.redis_conn.sadd(self.redis_key, url):
                yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail, meta={"item":copy.deepcopy(item)})
            # break
        next_href = html.xpath("//a[@class='next']/@href")
        if next_href:
            next_url = "https://www.cib.com.cn" + next_href[0]
            print(next_url)
            yield scrapy.Request(url=next_url, dont_filter=True, callback=self.parse)

    def get_detail(self, response):
        resp_url = response.url
        item = response.meta["item"]
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        p_list = html.xpath("//div[@class='middle']/p")
        front_url = re.findall("(.*\/)", resp_url)[0]
        htmlContent = ""
        content = ""
        for p in p_list:
            p_htmlContent = etree.tostring(p, encoding='utf-8', method='html', pretty_print=True).decode('utf-8')
            p_htmlContent = p_htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
            htmlContent = htmlContent + p_htmlContent
            p_content = p.xpath(".//text()")[0]
            content = content + p_content + "\n"
        item["content"] = content
        item["htmlContent"] = htmlContent
        yield item
