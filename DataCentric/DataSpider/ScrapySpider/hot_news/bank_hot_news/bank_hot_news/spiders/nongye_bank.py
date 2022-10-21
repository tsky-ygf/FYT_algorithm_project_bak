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

class NongyeBankSpider(scrapy.Spider):
    name = 'nongye_bank'
    start_urls = ['https://www.abchina.com/cn/AboutABC/nonghzx/NewsCenter/default.htm']
    redis_key = "banking_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)

    def parse(self, response):
        resp_url = response.url
        text = response.body.decode("utf-8")
        yield scrapy.Request(url=resp_url, dont_filter=True, callback=self.get_detail_url)
        max_page = re.findall("countPage = (.*?);",text)[0]
        for i in range(1,int(max_page)):
            url = resp_url.replace("default.htm",f"default_{i}.htm")
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_url)


    def get_detail_url(self, response):
        resp_url = response.url
        item = BankHotNewsItem()
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        span_list = html.xpath("//li[@class='cf']/span")
        print(f"get_detail_url start")
        for span in span_list:
            href = span.xpath("./a/@href")[0]
            title = span.xpath("./a/text()")[0]
            url = href.replace("./","https://www.abchina.com/cn/AboutABC/nonghzx/NewsCenter/")
            pubDate = re.sub("\s","","".join(span.xpath("./span/text()")))
            item["pubDate"] = pubDate
            item["title"] = "【中国农业银行】" + title
            if self.redis_conn.sadd(self.redis_key, url):
                yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail, meta={"item":copy.deepcopy(item)})
            # break

    def get_detail(self, response):
        item = response.meta["item"]
        resp_url = response.url
        uq_id = encode_md5(resp_url)
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        front_url = re.findall("(.*\/)", resp_url)[0]
        div = html.xpath("//div[@class='TRS_Editor']")[0]
        htmlContent = etree.tostring(div, encoding="utf-8", method="html", pretty_print=True).decode("utf-8")
        htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
        content = div.xpath(".//text()")
        content = "".join(content)
        item["uq_id"] = uq_id
        item["url"] = resp_url
        item["content"] = content
        item["htmlContent"] = htmlContent
        item["source"] = "中国农业银行"
        item["category"] = "金融银行"
        yield item

