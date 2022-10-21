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

class GongshangSpider(scrapy.Spider):
    name = 'gongshang'
    start_urls = ['http://member.icbc.com.cn/ICBC/%e5%b7%a5%e8%a1%8c%e9%a3%8e%e8%b2%8c/%e5%b7%a5%e8%a1%8c%e5%bf%ab%e8%ae%af/default.htm']
    redis_key = "banking_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)

    def parse(self, response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        href_list = html.xpath("//td[@align='left']/span/a/@href")
        for href in href_list:
            url = "http://member.icbc.com.cn" + href
            # if self.redis_conn.sadd(self.redis_key, next_url):
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail)
            # break
        next_page = html.xpath("//a[@class='textgs']/@href")
        if next_page:
            next_url = "http://member.icbc.com.cn" + next_page[0]
            yield scrapy.Request(url=next_url, dont_filter=True, callback=self.parse)

    def get_detail(self, response):
        item = BankHotNewsItem()
        resp_url = response.url
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        front_url = re.findall("(.*\/)", resp_url)[0]
        title = re.findall("META name=\"ICBCPostingTitle\" content=\"(.*?)\"",text)[0]
        pubDate = re.findall("name=\"ICBCPostingDate\" content=\"(.*?)\"",text)[0]
        span = html.xpath("//span[@id='MyFreeTemplateUserControl']")[0]
        htmlContent = etree.tostring(span, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
        content = "".join(span.xpath(".//text()"))
        item['url'] = resp_url
        item['uq_id'] = encode_md5(resp_url)
        item['title'] = "【中国工商银行】" + title
        item['htmlContent'] = htmlContent
        item['content'] = content
        item['pubDate'] = pubDate
        item['source'] = "中国工商银行"
        item['category'] = "金融银行"
        yield item
