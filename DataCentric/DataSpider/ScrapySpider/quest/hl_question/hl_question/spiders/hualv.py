import copy
import hashlib
import re
import redis
import scrapy
from lxml import etree
import logging
from hl_question.items import HlQuestionItem


def encode_md5(str):
    if str:
        hl = hashlib.md5()
        hl.update(str.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class HualvSpider(scrapy.Spider):
    name = 'hualv'
    allowed_domains = ['www.baidu.com']
    start_urls = ["https://www.66law.cn/question/answer/"]
    redis_key = "hl_question"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)

    def parse(self, response):
        logging.info(f"method:parse -->> {response.url}")
        item = HlQuestionItem()
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        a_list = html.xpath("//div[@class='ovh']/ul/li/a")[1:]
        for a in a_list:
            href = a.xpath("./@href")[0]
            url = "https://www.66law.cn" + href
            first_type = re.sub("\s","","".join(a.xpath("./text()")))
            item['first_type'] = first_type
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_second_type, meta={"item": copy.deepcopy(item)})
            # break

    def get_second_type(self, response):
        logging.info(f"method:get_second_type -->> {response.url}")
        item = response.meta["item"]
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        a_list = html.xpath("//div[@class='ovh']/ul/li/a")[1:]
        for a in a_list:
            href = a.xpath("./@href")[0]
            url = "https://www.66law.cn" + href
            second_type = re.sub("\s", "", "".join(a.xpath("./text()")))
            item['second_type'] = second_type
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_page_url, meta={"item": copy.deepcopy(item)})
            # break

    def get_page_url(self, response):
        logging.info(f"method:get_page_url -->> {response.url}")
        resp_url = response.url
        item = response.meta["item"]
        for i in range(500,6000):
            url = re.sub("list\d+\.aspx",f"list{i}.aspx",resp_url)
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_url, meta={"item": copy.deepcopy(item)})
            break

    def get_detail_url(self, response):
        logging.info(f"method:get_detail_url -->> {response.url}")
        item = response.meta["item"]
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        href_list = html.xpath("//div[@class='title']/a/@href")
        if not href_list:
            return
        for href in href_list:
            url = "https://www.66law.cn" + href
            item["url"] = url
            item["uq_id"] = encode_md5(item["second_type"] + url)
            if self.redis_conn.sadd(self.redis_key, item["uq_id"]):
                yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail, meta={"item": copy.deepcopy(item)})
            # yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail, meta={"item": copy.deepcopy(item)})
            # break

    def get_detail(self, response):
        logging.info(f"method:get_detail -->> {response.url}")
        item = response.meta["item"]
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        question = html.xpath("//div[@class='tit-bar']/h1/text()")[0]
        item["question"] = question
        item["source"] = "华律网"
        pubDate = re.findall("pubDate\":\"(.*?)T",text)[0]
        item["pubDate"] = pubDate
        da_item = html.xpath("//div[contains(@class,'da-item')]")[0]
        answer = da_item.xpath("./p//text()")
        answer = "".join(answer)
        item["answer"] = answer
        content = da_item.xpath("./p")[0]
        htmlContent = etree.tostring(content, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        item["htmlContent"] = htmlContent
        yield item

# if __name__ == '__main__':
#     url = "https://www.66law.cn/question/answer/27-list1.aspx"
#     url = re.sub("list\d+\.aspx","list188.aspx",url)
#     print(url)
