import copy
import hashlib
import re

import scrapy
from lxml import etree
from quest.items import QuestItem

class BankingQuestionSpider(scrapy.Spider):
    name = 'banking_question'
    start_urls = [
        # 'https://china.findlaw.cn/jingjifa/yinghanfa/',
        'https://www.66law.cn/zhinan/jinrongfuwu/',
    ]

    def start_requests(self):
        for url in self.start_urls:
            if "https://china.findlaw.cn" in url:
                yield scrapy.Request(url= url,callback=self.zfw_parse)
            if "www.66law.cn" in url:
                yield scrapy.Request(url=url,callback=self.hl_parse)

    def zfw_parse(self, response):
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        menu_list = html.xpath("//div[@class='menu-head']")[1:]
        for menu in menu_list:
            head_name = menu.xpath("./a/text()")
            href = menu.xpath("./a/@href")[0]
            # print(head_name)
            # print(href)
            yield scrapy.Request(url=href,callback=self.zfw_get_more)
        # print(text)

    def zfw_get_more(self,response):
        item = QuestItem()
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        div_list = html.xpath("//div[contains(@class,'common-marry-family')]")
        for div in div_list:
            question_type = div.xpath(".//a[@class='common-title-style4']//text()")[1]
            item['question_type'] = question_type
            more_href = div.xpath(".//a[@class='common-load-more']/@href")[0]
            yield scrapy.Request(url=more_href,callback=self.zfw_get_detail_url,meta={'item':copy.deepcopy(item)})

    def zfw_get_detail_url(self,response):
        print(f"zfw_get_detail_url start, url: {response.url} \n *********************************************")
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        more_flag = html.xpath(".//a[@class='common-load-more']/@href")
        if more_flag:
            for href in more_flag:
                print(f"have more href, url: {href}")
                yield scrapy.Request(url=href,callback=self.zfw_get_more)
        else:
            href_list = html.xpath("//div[@class='model-classify-list']/ul/li//a/@href")
            for href in href_list:
                print(href)
                yield scrapy.Request(url=href,callback=self.zfw_get_detail,meta={'item':copy.deepcopy(item)})
        print(f"zfw_get_detail over \n *********************************************")

    def zfw_get_detail(self,response):
        print(f"zfw_get_detail start, url: {response.url}")
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        txt = html.xpath("//div[@class='article']//text()")
        # print(txt)
        if txt:
            txt = self.clear_front(txt)
            answer = "".join(txt)
            answer = re.sub(" |&nbsp;","",answer)
            item['answer'] = answer
            url = response.url
            item['url'] = url
            item['source'] = "找法网"
            item['model_type'] = "银行金融"
            pubDate = re.findall("pubDate\":\"(.*?)\"",text)[0]
            item['pubData'] = pubDate[:10]
            question = re.findall("title\": \"(.*?)\"",text)[0]
            item['question'] = question
            item['uq_id'] = self.encode_md5(url + question + answer)
            yield item

    def hl_parse(self,response):
        item = QuestItem()
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        div_list = html.xpath("//div[@class='f20-nav']")
        for div in div_list:
            quest_type = div.xpath("./strong/text()")[0]
            item['question_type'] = quest_type
            more_href = div.xpath("./a/@href")[0]
            more_url = "https://www.66law.cn" + more_href
            yield scrapy.Request(url=more_url,callback=self.hl_get_page,meta={"item":copy.deepcopy(item)})

    def hl_get_page(self,response):
        item = response.meta["item"]
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        # page_3.aspx
        front_url = response.url
        max_page = html.xpath("//div[contains(@class,'u-page')]/a/text()")[-2]
        for page in range(1,int(max_page)+1):
            url = front_url + f"page_{page}.aspx"
            print(url)
            yield scrapy.Request(url=url,callback=self.hl_get_detail_url,meta={"item":copy.deepcopy(item)})
            # break

    def hl_get_detail_url(self,response):
        item = response.meta["item"]
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        href_list = html.xpath("//a[@class='img']/@href")
        for href in href_list:
            url = "https://www.66law.cn" + href
            yield scrapy.Request(url=url,callback=self.hl_get_detail,meta={"item":copy.deepcopy(item)})
            # break

    def hl_get_detail(self,response):
        item = response.meta["item"]
        print(item)
        print(response.url)
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        question = html.xpath("//div[@class='det-title']/h1/text()")[0]
        source = "华律网"
        pubDate = html.xpath("//div[@class='det-title']/div[@class='info']/span[2]/text()")[0]
        answer = html.xpath("//div[@class='det-nr']//text()")
        answer = "".join(answer)
        content = html.xpath("//div[@class='det-nr']")[0]
        txt = etree.tostring(content, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        content = "\n".join(re.findall("<p.*?2em.*?>(.*?)</p>", txt))
        if content:
            item['answer'] = re.sub("<.*?>", "", content)
            url = response.url
            item['url'] = url
            item['question'] = question
            item['source'] = source
            item['pubData'] = pubDate
            # item['answer'] = answer            item['url'] = url
            item['model_type'] = "银行金融"
            item['uq_id'] = self.encode_md5(url + question + answer)
            yield item

    def clear_front(self,txt_list):
        index = 0
        while index < len(txt_list):
            if re.sub("\s", "", txt_list[index]):
                break
            index += 1
        return txt_list[index:]


    def encode_md5(self,str):
        if str:
            hl = hashlib.md5()
            hl.update(str.encode(encoding='utf-8'))
            return hl.hexdigest()
        return ''



# question = scrapy.Field()
#     # 唯一主键
#     uq_id = scrapy.Field()
#     # 来源
#     source = scrapy.Field()
