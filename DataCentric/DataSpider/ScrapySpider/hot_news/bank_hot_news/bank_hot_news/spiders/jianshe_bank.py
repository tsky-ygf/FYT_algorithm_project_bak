import hashlib
import re
import scrapy
from lxml import etree
import redis
from bank_hot_news.items import BankHotNewsItem


def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class JiansheBankSpider(scrapy.Spider):
    name = 'jianshe_bank'
    redis_key = "banking_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    start_urls = ['http://www3.ccb.com/cn/v3/include/notice/zxgg_1.html']


    def parse(self,response):
        text = response.body.decode('utf-8')
        page = re.findall("共(.*?)页",text)[0]
        for i in range(1,int(page)+1):
            url = f"http://www3.ccb.com/cn/v3/include/notice/zxgg_{i}.html"
            yield scrapy.Request(url=url,dont_filter=True,callback=self.get_detail_url)

    def get_detail_url(self, response):
        item = BankHotNewsItem()
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        a_list = html.xpath("//ul[@class='list']/li/a")
        for a in a_list:
            title = a.xpath("./@title")[0]
            item['title'] = title
            href = a.xpath("./@href")[0]
            url = href.replace("./","http://www2.ccb.com/cn/v3/include/notice/")
            res = self.redis_conn.sadd(self.redis_key, url)
            if res == 1:
                yield scrapy.Request(url=url,dont_filter=True,callback=self.get_detail)

    def get_detail(self,response):
        item = BankHotNewsItem()
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        content = html.xpath("//div[contains(@class,'f14')]")[0]
        txt = etree.tostring(content, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        url = response.url
        title = content.xpath("./h2[@class='Yahei']/text()")[0]
        pubDate = re.findall("发布时间：(.*?)<",txt)[0]
        item['htmlContent'] = txt
        item['content'] = "".join(content.xpath("./div[@id='ti']//text()"))
        item['url'] = url
        item['title'] = "【中国建设银行】" + title
        item['source'] = "中国建设银行"
        item['category'] = "金融银行"
        item['pubDate'] = pubDate
        item['uq_id'] = encode_md5(url)
        yield item


