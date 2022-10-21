import hashlib

import scrapy
import re
from lxml import etree
import redis
from sfj_hot_news.items import SfjHotNewsItem

class ZhejiangSpider(scrapy.Spider):
    # 浙江省司法厅 新闻
    name = 'zhejiang'
    redis_key = "sfj_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    start_urls = [
        'http://zhejiang.chinatax.gov.cn/molsdule/web/jpage/dataproxy.jsp?startrecord=361&endrecord=405&perpage=15']
    data = {
        "col": "1",
        "appid": "1",
        "webid": "2747",
        "path": "/",
        "columnid": "1229671105",
        "sourceContentType": "1",
        "unitid": "6325023",
        "webname": "浙江省司法厅",
        "permissiontype": "0"
    }

    def get_start_urls(self):
        url_list = []
        for i in range(3):
            s = i * 45 + 1
            o = s + 44
            # url = f'http://zhejiang.chinatax.gov.cn/module/web/jpage/dataproxy.jsp?startrecord={s}&endrecord={o}&perpage=15'
            url = f"https://sft.zj.gov.cn/module/jpage/dataproxy.jsp?startrecord={s}&endrecord={o}&perpage=14"
            url_list.append(url)
        return url_list

    def start_requests(self):
        for url in self.get_start_urls():
            yield scrapy.FormRequest(url=url, callback=self.parse, formdata=self.data)

    def parse(self, response):
        text = response.body.decode("utf-8")
        # print(text)
        href_list = re.findall('href=\'(.*?)\'',text)
        print(href_list)
        if href_list:
            for href in href_list:
                if "http" not in href:
                    url = "https://sft.zj.gov.cn" + href
                    res = self.redis_conn.sadd(self.redis_key, url)
                    if res == 1:
                        print(url)

                        yield scrapy.Request(url=url,dont_filter=True,callback=self.get_detail)
        else:
            print(f"no href,url:{response.url}")
        # http://zhejiang.chinatax.gov.cn/module/web/jpage/dataproxy.jsp?startrecord=91&endrecord=135&perpage=15

    def get_detail(self,response):
        item = SfjHotNewsItem()
        url = response.url
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        content = html.xpath("//div[@class='nr']")[0]
        txt = etree.tostring(content, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        txt = txt.replace("href=\"/picture", "href=\"https://sft.zj.gov.cn/picture").replace("href=\'/picture",
                                                                                       "href=\'https://sft.zj.gov.cn/picture").replace(
            "src=\"/picture", "src=\"https://sft.zj.gov.cn/picture").replace("src=\'/picture",
                                                                             "src=\'https://sft.zj.gov.cn/picture")
        item['htmlContent'] = txt
        item['url'] = url
        # item['pubDate'] = re.findall("发布时间：(.*?) ",text)[0]
        item['pubDate'] = html.xpath("//meta[@name='PubDate']/@content")[0][:10]
        item['province'] = '浙江'
        content = "".join(re.findall("<p.*?2em.*?>(.*?)</p>",txt))
        if content:
            item['content'] = re.sub("<.*?>","",content)
        item['title'] = html.xpath("//meta[@name='ArticleTitle']/@content")[0]
        item['uq_id'] = self.encode_md5(url)
        # print(item)
        yield item

    def encode_md5(self, str):
        if str:
            hl = hashlib.md5()
            hl.update(str.encode(encoding='utf-8'))
            return hl.hexdigest()
        return ''