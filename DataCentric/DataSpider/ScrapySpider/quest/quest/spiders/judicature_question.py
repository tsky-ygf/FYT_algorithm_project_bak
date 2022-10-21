import hashlib
import re

import scrapy
from lxml import etree
from quest.items import QuestItem

class JudicatureQuestionSpider(scrapy.Spider):
    name = 'judicature_question'
    start_urls = ['http://www.moj.gov.cn/pub/sfbgw/zwfw/zwfwbszn/bsznlsfw/']

    def parse(self, response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        href_list = html.xpath("//li[@class='twoList']/a/@href")[1:]
        url_list = [response.url]
        for href in href_list:
            href = re.sub("\.\.","",href)
            url = "http://www.moj.gov.cn/pub/sfbgw/zwfw/zwfwbszn" + href
            url_list.append(url)
        for url in url_list:
            yield scrapy.Request(url=url,callback=self.sfb_get_page)
            # break
        # text = response.body.decode("//li[@class='twoList']/a/@href")

    def sfb_get_page(self,response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        url = response.url
        max_page = re.findall("createPageHTML\((.*?), ", text)[0]
        href_list = html.xpath("//a[@class='leftA']/@href")
        # print(href_list)
        for href in href_list:
            yield scrapy.Request(url=href, callback=self.sfb_get_detail)
        if max_page and max_page != '1':
            for index in range(1,int(max_page)):
                tmp_url = url + f"index_{index}.html"
                yield scrapy.Request(url=tmp_url, callback=self.sfb_get_detail_url)

    def sfb_get_detail_url(self, response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        href_list = html.xpath("//a[@class='leftA']/@href")
        for href in href_list:
            yield scrapy.Request(url=href, callback=self.sfb_get_detail)

    def sfb_get_detail(self, response):
        item = QuestItem()
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        url = response.url
        question = html.xpath("//h1[@class='phone_size1']/text()")[0]
        item['question'] = question
        pubData = re.findall("发布时间：(.*?) ", text)[0]
        item["pubData"] = pubData
        content = html.xpath("//div[@class='TRS_Editor']")[0]
        txt = etree.tostring(content, encoding="utf-8", pretty_print=True, method="html").decode('utf-8')
        content = "".join(re.findall("<p.*?2em.*?>(.*?)</p>", txt))
        question_type = html.xpath("//a[@class='news_content_style']/@title")[-1]
        model_type = "司法局"
        source = "中华人民共和国司法部"
        if content:
            answer = re.sub("<.*?>", "", content)
            item['answer'] = answer
            item['url'] = url
            item['question_type'] = question_type
            item['model_type'] = model_type
            item['source'] = source
            item['uq_id'] = self.encode_md5(url+question+answer)
            yield item

    def encode_md5(self,str):
        if str:
            hl = hashlib.md5()
            hl.update(str.encode(encoding='utf-8'))
            return hl.hexdigest()
        return ''
