import hashlib
import re
import redis
import scrapy
from lxml import etree
from scjd_hot_news.items import ScjdHotNewsItem

def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''
class ZhejiangSpider(scrapy.Spider):
    name = 'zhejiang'
    redis_key = "scjd_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    start_urls = ["http://zjamr.zj.gov.cn/module/jpage/dataproxy.jsp?startrecord=90&endrecord=135&perpage=15"]
    # http://zjamr.zj.gov.cn/module/jpage/dataproxy.jsp?startrecord=46&endrecord=90&perpage=15
    data = {
        "col": "1",
        "appid": "1",
        "webid": "3397",
        "path": "/",
        "columnid": "1228969893",
        "sourceContentType": "1",
        "unitid": "5324824",
        "webname": "浙江省市场监督管理局（浙江省知识产权局）",
        "permissiontype": "0"
    }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.FormRequest(url=url,formdata=self.data,dont_filter=True,callback=self.zj_parse_page)

    def zj_parse_page(self, response, **kwargs):
        text = response.body.decode("utf-8")
        totalPage = re.findall("totalpage>(.*?)</totalpage",text)[0]
        page = int(int(totalPage)/3 + 1)
        for i in range(page):
            s = i * 45 + 1
            o = i * 45 + 45
            url = f"http://zjamr.zj.gov.cn/module/jpage/dataproxy.jsp?startrecord={s}&endrecord={o}&perpage=15"
            yield scrapy.FormRequest(url=url, formdata=self.data, callback=self.zj_get_detail_url)

    def zj_get_detail_url(self, response, **kwargs):
        text = response.body.decode("utf-8")
        href_list = re.findall("href=\"(.*?)\"",text)
        for href in href_list:
            if "http" in href:
                continue
            url = "http://zjamr.zj.gov.cn" + href
            res = self.redis_conn.sadd(self.redis_key, url)
            if res == 1:
                yield scrapy.Request(url=url,dont_filter=True,callback=self.zj_get_detail)

    def zj_get_detail(self, response, **kwargs):
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        title = html.xpath(".//title/text()")[0]
        pubDate = html.xpath("//meta[@name='PubDate']/@content")[0][:10]
        url = response.url
        content_div = html.xpath("//div[@class='nr_text']")[0]
        htmlContent = etree.tostring(content_div, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        htmlContent = htmlContent.replace("href=\"/picture", "href=\"http://zjamr.zj.gov.cn/picture").replace("href=\'/picture",
                                                                                             "href=\'http://zjamr.zj.gov.cn/picture").replace(
            "src=\"/picture", "src=\"http://zjamr.zj.gov.cn/picture").replace("src=\'/picture",
                                                                             "src=\'https://sft.zj.gov.cn/picture")

        content = "".join(re.findall("<p.*?2em.*?>(.*?)</p>", htmlContent))
        if content:
            item = ScjdHotNewsItem()
            item['uq_id'] = encode_md5(url)
            item['title'] = "【浙江省】" + title
            item['content'] = re.sub("<.*?>","",content.replace("<br>","\n"))
            item['htmlContent'] = htmlContent
            item['province'] = "浙江"
            item['pubDate'] = pubDate
            item['url'] = url
            item['source'] = "浙江省市场监督管理局（浙江省知识产权局）"
            item['category'] = "市场监督"
            item['question_type'] = "市场监督动态"
            yield item
