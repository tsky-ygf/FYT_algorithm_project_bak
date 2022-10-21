import copy
import hashlib
import re
from wl_hot_news.items import WlHotNewsItem
import scrapy
import redis
from lxml import etree


def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class ZhejiangSpider(scrapy.Spider):
    name = 'zhejiang'
    start_urls = [
        'http://ct.zj.gov.cn/col/col1652990/index.html',
        'http://ct.zj.gov.cn/col/col1652991/index.html',
        'http://ct.zj.gov.cn/col/col1652992/index.html',
    ]
    redis_key = "wl_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)


    def start_requests(self):
        format_url = "http://ct.zj.gov.cn/module/jpage/dataproxy.jsp?startrecord=1&endrecord=45&perpage=15"
        data90 = {
            "col": "1",
            "appid": "1",
            "webid": "3214",
            "path": "^%^2F",
            "columnid": "1652990",
            "sourceContentType": "1",
            "unitid": "7814811",
            "webname": "^%^E6^%^B5^%^99^%^E6^%^B1^%^9F^%^E7^%^9C^%^81^%^E6^%^96^%^87^%^E5^%^8C^%^96^%^E5^%^92^%^8C^%^E6^%^97^%^85^%^E6^%^B8^%^B8^%^E5^%^8E^%^85",
            "permissiontype": "0"
        }
        data91 = {
            "col": "1",
            "appid": "1",
            "webid": "3214",
            "path": "^%^2F",
            "columnid": "1652991",
            "sourceContentType": "1",
            "unitid": "7814811",
            "webname": "^%^E6^%^B5^%^99^%^E6^%^B1^%^9F^%^E7^%^9C^%^81^%^E6^%^96^%^87^%^E5^%^8C^%^96^%^E5^%^92^%^8C^%^E6^%^97^%^85^%^E6^%^B8^%^B8^%^E5^%^8E^%^85",
            "permissiontype": "0"
        }
        data92 = {
            "col": "1",
            "appid": "1",
            "webid": "3214",
            "path": "^%^2F",
            "columnid": "1652992",
            "sourceContentType": "1",
            "unitid": "7814811",
            "webname": "^%^E6^%^B5^%^99^%^E6^%^B1^%^9F^%^E7^%^9C^%^81^%^E6^%^96^%^87^%^E5^%^8C^%^96^%^E5^%^92^%^8C^%^E6^%^97^%^85^%^E6^%^B8^%^B8^%^E5^%^8E^%^85",
            "permissiontype": "0"
        }
        for url in self.start_urls:
            if "2990" in url:
                yield scrapy.FormRequest(url=format_url, dont_filter=True, callback=self.get_page_parse, formdata=data90, meta={"data":copy.deepcopy(data90), "question_type":"浙江文旅"})
            if "2991" in url:
                yield scrapy.FormRequest(url=format_url, dont_filter=True, callback=self.get_page_parse, formdata=data91, meta={"data":copy.deepcopy(data91), "question_type":"文旅热点"})
            if "2992" in url:
                yield scrapy.FormRequest(url=format_url, dont_filter=True, callback=self.get_page_parse, formdata=data92, meta={"data":copy.deepcopy(data92), "question_type":"地方动态"})
            break

    def get_page_parse(self, response):
        text = response.body.decode("utf-8")
        question_type = response.meta["question_type"]
        page = re.findall("<totalpage>(.*?)</totalpage>", text)[0]
        max_page = int(int(page)/3) + 1
        data = response.meta["data"]
        for i in range(max_page):
            s = i * 45 + 1
            o = i * 45 + 45
            url = f"http://ct.zj.gov.cn/module/jpage/dataproxy.jsp?startrecord={s}&endrecord={o}&perpage=15"
            yield scrapy.FormRequest(url=url, dont_filter=True, callback=self.get_detail_url, formdata=data, meta={"question_type":question_type})
            # break

    def get_detail_url(self, response):
        item = WlHotNewsItem()
        text = response.body.decode("utf-8")
        question_type = response.meta["question_type"]
        data_list = re.findall("<li><a href=\"(.*?)\" target=\"_blank\" title=\".*?\">(.*?)</a><span>(.*?)</span>.*?</record>",text)
        for data_item in data_list:
            href = data_item[0]
            title = data_item[1]
            pubDate = data_item[2]
            if re.match("^art",href):
                url = "http://ct.zj.gov.cn/" + href
                item['url'] = url
                item['uq_id'] = encode_md5(url)
                item['title'] = "【浙江省】" + title
                item['pubDate'] = pubDate
                item['category'] = "文旅专栏"
                item['question_type'] = question_type
                # print(item)
                yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_artType, meta={"item":copy.deepcopy(item)})
            elif re.match("^https://www.mct.gov.cn",href):
                url = href
                item['url'] = url
                item['uq_id'] = encode_md5(url)
                item['title'] = "【浙江省】" + title
                item['pubDate'] = pubDate
                item['category'] = "文旅专栏"
                item['question_type'] = question_type
                yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_mctType, meta={"item":copy.deepcopy(item)})
            else:
                continue

    def get_detail_artType(self, response):
        text = response.body.decode("utf-8")
        item = response.meta["item"]
        html = etree.HTML(text)
        div = html.xpath("//div[@class='main_section']")[0]
        htmlContent = etree.tostring(div, encoding='utf-8', pretty_print=True, method='html').decode('utf-8')
        htmlContent = htmlContent.replace("src=\"/picture", "src=\"http://ct.zj.gov.cn/picture")
        content = "".join(div.xpath(".//text()"))
        item["content"] = content
        item["htmlContent"] = htmlContent
        item["province"] = "浙江省"
        item["source"] = "浙江省文化和旅游厅"
        yield item

    def get_detail_mctType(self, response):
        text = response.body.decode("utf-8")
        item = response.meta["item"]
        html = etree.HTML(text)
        div = html.xpath("//div[@class='TRS_Editor']")[0]
        htmlContent = etree.tostring(div, encoding='utf-8', pretty_print=True, method='html').decode('utf-8')
        # htmlContent = htmlContent.replace("src=\"/picture", "src=\"http://ct.zj.gov.cn/picture")
        resp_url = response.url
        front_url = re.findall("(.*\/)", resp_url)[0]
        htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
        content = "".join(div.xpath(".//text()"))
        item["content"] = content
        item["htmlContent"] = htmlContent
        item["province"] = "浙江省"
        item["source"] = "浙江省文化和旅游厅"
        yield item
