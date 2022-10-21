import hashlib
import re
from lxml import etree
import scrapy
import redis
from jt_hot_news.items import JtHotNewsItem

def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class ZgjtbSpider(scrapy.Spider):
    name = 'zgjtb'
    allowed_domains = ['www.baidu.com']
    start_urls = ['https://www.mot.gov.cn/jiaotongyaowen/']
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    redis_key = "jt_hot_news"

    def parse(self, response):
        text = response.body.decode("utf-8")
        resp_url = response.url
        max_page = re.findall("createPageHTML\((.*?), 0, \"index\", \"html\"",text)[0]
        yield scrapy.Request(url=resp_url, dont_filter=True, callback=self.get_detail_url)
        for i in range(1,int(max_page)):
            url = resp_url + f"index_{i}.html"
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_url)

    def get_detail_url(self, response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        href_list = html.xpath("//div[@role='tabpanel']/a/@href")
        for href in href_list:
            url = href.replace("./","https://www.mot.gov.cn/jiaotongyaowen/")
            if self.redis_conn.sadd(self.redis_key,url):
                yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail)

    def get_detail(self, response):
        item = JtHotNewsItem()
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        resp_url = response.url
        title = html.xpath("//meta[@name='ArticleTitle']/@content")[0]
        pubDate = html.xpath("//meta[@name='PubDate']/@content")[0][:10]
        div = html.xpath("//div[contains(@class,'trs_paper_default')]")[0]
        htmlContent = etree.tostring(div, encoding='utf-8', method='html', pretty_print=True).decode('utf-8')
        front_url = re.findall("(.*\/)", resp_url)[0]
        htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
        # htmlContent = html.xpath("//div[@id='Zoom']//text()")
        content = "\n".join(div.xpath(".//text()"))
        # content = "\n".join(re.findall("<p.*?2em.*?>(.*?)</p>", htmlContent))
        # content = re.sub("<.*?>", "", content)
        # print(content)
        item["url"] = resp_url
        item["uq_id"] = encode_md5(resp_url)
        item["title"] = title
        item["content"] = content
        item["htmlContent"] = htmlContent
        item["pubDate"] = pubDate
        item["source"] = "中华人民共和国交通运输部"
        item["category"] = "交通专栏"
        item["question_type"] = "交通要闻"
        yield item