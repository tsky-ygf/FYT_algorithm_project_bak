import copy
import hashlib
import re
from pprint import pprint
import redis

import scrapy
from lxml import etree
from swj_hot_news.items import SwjHotNewsItem

class ZhejiangSpider(scrapy.Spider):
    name = 'zhejiang'
    redis_key = "swj_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    start_urls = ['http://zhejiang.chinatax.gov.cn/module/web/jpage/dataproxy.jsp?startrecord=361&endrecord=405&perpage=15']
    data = {
        "col": "1",
        "appid": "1",
        "webid": "15",
        "path": "/",
        "columnid": "13226",
        "sourceContentType": "1",
        "unitid": "57907",
        "webname": "国家税务总局浙江省税务局",
        "permissiontype": "0"
    }
    def get_start_urls(self):
        url_list = []
        for i in range(15):
            s = i * 45 + 1
            o = s + 44
            url = f'http://zhejiang.chinatax.gov.cn/module/web/jpage/dataproxy.jsp?startrecord={s}&endrecord={o}&perpage=15'
            url_list.append(url)
        return url_list

    def start_requests(self):
        for url in self.get_start_urls():
            yield scrapy.FormRequest(url=url, dont_filter=True, callback=self.parse,formdata=self.data)

    def parse(self, response):
        text = response.body.decode("utf-8")
        href_list = re.findall('href=\"(.*?)\"',text)
        # page = response.meta['nextPage']
        # next_page = page + 1
        if href_list:
            # print(f"page:{next_page}")
            # next_url = f"http://zhejiang.chinatax.gov.cn/module/web/jpage/dataproxy.jsp?startrecord={page * 45 + 1}&endrecord={page * 45 + 45}&perpage=15'"
            # yield scrapy.FormRequest(url=next_url,formdata=self.data, dont_filter=True, callback=self.parse,meta={'nextPage':next_page})
            for href in href_list:
                if "http" not in href:
                    url = "http://zhejiang.chinatax.gov.cn" + href
                    res = self.redis_conn.sadd(self.redis_key, url)
                    if res == 1:
                        yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail)
        else:
            print(f"no href,url:{response.url}")
        # http://zhejiang.chinatax.gov.cn/module/web/jpage/dataproxy.jsp?startrecord=91&endrecord=135&perpage=15

    def get_detail(self,response):
        item = SwjHotNewsItem()
        url = response.url
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        content = html.xpath("//div[@class='info-cont']")[0]
        txt = etree.tostring(content, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        item['url'] = url
        # item['htmlContent'] = txt.replace("href=\"/picture","href=\"http://zhejiang.chinatax.gov.cn/picture")
        item['htmlContent'] = txt.replace("href=\"/picture","href=\"http://zhejiang.chinatax.gov.cn/picture").replace("href=\'/picture","href=\'http://zhejiang.chinatax.gov.cn/picture").replace("src=\"/picture","src=\"http://zhejiang.chinatax.gov.cn/picture").replace("src=\'/picture","src=\'http://zhejiang.chinatax.gov.cn/picture")
        item['pubDate'] = re.findall("发布时间：(.*?) ",text)[0]
        item['province'] = '浙江'
        item['source'] = '浙江税务局'
        item['category'] = '税务'
        content = "".join(re.findall("<p.*?2em.*?>(.*?)</p>",txt))
        if content:
            item['content'] = re.sub("<.*?>","",content)
        item['title'] = html.xpath("//meta[@name='ArticleTitle']/@content")[0]
        item['uq_id'] = self.encode_md5(url)
        yield item


    def encode_md5(self, str):
        if str:
            hl = hashlib.md5()
            hl.update(str.encode(encoding='utf-8'))
            return hl.hexdigest()
        return ''
