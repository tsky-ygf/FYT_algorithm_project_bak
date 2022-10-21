import copy
import hashlib
import re

from lxml import etree
import scrapy
import json
from quest.items import QuestItem


class QuestandanswerSpider(scrapy.Spider):
    name = 'questAndAnswer'
    start_urls = [
        # 12366纳税服务平台
        "https://12366.chinatax.gov.cn/znzx/znzx/getZnwdRdwtType",
        # # 律图
        "https://www.64365.com/ask/browse_c282/",
        "https://www.64365.com/ask/browse_c283/",
        # # 法律快车
        "https://www.lawtime.cn/askzt/listview_22.html",
        # # 找法网
        "https://china.findlaw.cn/jingjifa/caishuifa/",
        # 华律 直通车
        "https://www.66law.cn/zhinan/qiyebanshi/qiyenashui/",
    ]

    def start_requests(self):
        for url in self.start_urls:
            if url == "https://12366.chinatax.gov.cn/znzx/znzx/getZnwdRdwtType":
                yield scrapy.Request(url=url,callback=self.parse)
            elif "www.64365.com" in url:
                yield scrapy.Request(url=url,callback=self.lvtu_parse)
            elif "china.findlaw.cn" in url:
                yield scrapy.Request(url=url,callback=self.zfw_parse)
            elif "www.lawtime.cn" in url:
                yield scrapy.Request(url=url,callback=self.flkc_parse)
            elif "www.66law.cn" in url:
                yield scrapy.Request(url=url,callback=self.hualv_parse)

    def parse(self, response):
        item = QuestItem()
        text = response.body.decode("utf-8")
        json_data = json.loads(text)
        data_list = json_data['data']['data']
        for data_item in data_list:
            quest_type = data_item['typename']  # 问题类型
            item['question_type'] = quest_type
            quest_code = data_item['code']  # 问题code
            data = {
                "queryNum": "99",
                "wdfl": quest_code,
                "pageIndex": "1",
                "querySource": "1"
            }
            yield scrapy.FormRequest(url="https://12366.chinatax.gov.cn/znzx/znzx/getZnwdRdwt", formdata=data,
                                     meta={'item': copy.deepcopy(item)}, callback=self.get_question_code_parse)

    def get_question_code_parse(self,response):
        item = response.meta['item']
        text = response.body.decode('utf-8')
        json_data = json.loads(text)
        data_list = json_data['data']['data']
        for data_item in data_list:
            linkHdid = data_item['linkHdid']
            linkWtid = data_item['linkWtid']
            item['uq_id'] = self.encode_md5(linkHdid+linkWtid)
            data = {
                "answerId": linkHdid,  # linkHdid
                "answerTitleId": linkWtid,  # linkWtid
                "channel": "1",
                "answerTitle": "",
                "clickType": "0",
                "usertype": "",
                "yhId": "",
                "similarityIntegrate": "1"
            }
            yield scrapy.FormRequest(url="https://12366.chinatax.gov.cn/znzx/znzx/getQaAnswer",formdata=data,meta={'item':copy.deepcopy(item)},callback=self.get_answer_parse)

    def get_answer_parse(self,response):
        item = response.meta['item']
        text = response.body.decode('utf-8')
        json_data = json.loads(text)
        question = json_data['data']['answerTitle']  # 问题
        answer = json_data["data"]["ivrcontent"]  # 回答
        pubData = json_data['data']['gxsj']  # 发布日期
        question_sign = json_data['data']['zltypejc']  # 问题类型 标签
        hot_question_sign = json_data['data']['yjbq']  # 热点类型 标签
        item['url'] = response.url
        item['question'] = question
        item['answer'] = answer
        item['pubData'] = pubData[:10]
        item['question_sign'] = question_sign
        item['hot_question_sign'] = hot_question_sign
        item['model_type'] = "税务"
        item['source'] = "12366纳税服务平台"
        yield item

    def encode_md5(self,str):
        if str:
            hl = hashlib.md5()
            hl.update(str.encode(encoding='utf-8'))
            return hl.hexdigest()
        return ''

    def lvtu_parse(self,response):
        # 律图网 税务纠纷 问答数据
        # https://www.64365.com/ask/browse_c169/
        response_url = response.url
        item = QuestItem()
        item['question_type'] = "税务类纠纷"
        item['source'] = "律图"

        # item['hot_question_sign'] = ""
        # "https://www.64365.com/ask/browse_c282/",
        # "https://www.64365.com/ask/browse_c283/",

        if response_url == "https://www.64365.com/ask/browse_c282/":
            item["question_sign"] = "税务行政复议"
        if response_url == "https://www.64365.com/ask/browse_c283/":
            item["question_sign"] = "税务诉讼"
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        href_list = html.xpath("//div[contains(@class,'result_item')]/div/div/strong/a/@href")
        for href in href_list:
            new_url = "https://www.64365.com" + href
            yield scrapy.Request(url=new_url,callback=self.lvtu_get_ques_answ,meta={'item':copy.deepcopy(item)})
        next_flag = html.xpath("//div[contains(@class,'u-page')]/a[2]/@href")
        if next_flag:
            next_url = "https://www.64365.com" + next_flag[0]
            print(next_url)
            yield scrapy.Request(url=next_url,callback=self.lvtu_parse)

    def lvtu_get_ques_answ(self,response):
        # 律图网 解析详细内容
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        quest = html.xpath("//div[contains(@class,'problem_bar')]/h1/text()")
        item["question"] = quest[0] if quest else ''
        # /html/body/div[2]/div[2]/div[1]/div[1]/div[1]/span[4]
        pubData = html.xpath("//div[@class='info']/span[4]/text()")
        item["pubData"] = pubData[0][:10].replace(".","-") if pubData else ''
        answer_list = re.findall("<p class=\"tx\">(.*?)</p>",text)
        response_url = response.url
        item['url'] = response_url
        for answer in answer_list:
            answer = answer.replace("<br/>","\n")
            answer = re.sub("<.*?>","",answer)
            md5_str = response_url + item.get('question') + answer
            item['answer'] = answer
            item['uq_id'] = self.encode_md5(md5_str)
            item['model_type'] = "税务"
            yield item

    def zfw_parse(self,response):
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        a_list = html.xpath("//div[@id='j-tab-two']/a")
        href = a_list[0].xpath("./@href")[0]
        yield scrapy.Request(url=href,callback=self.get_zfw_question_type)

    def get_zfw_question_type(self,response):
        item = QuestItem()
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        menu_list = html.xpath("//div[@class='menu-item']")
        for menu in menu_list:
            question_type = menu.xpath("./div[@class='menu-head']/a/text()")[0]
            item["question_type"] = question_type
            item["source"] = "找法网"
            a_list = menu.xpath("./div[@class='menu-body']/div/a")
            for a in a_list:
                question_sign = a.xpath("./text()")[0]
                item["question_sign"] = question_sign
                href = a.xpath("./@href")[0]
                yield scrapy.Request(url=href,callback=self.get_zfw_question_list,meta={'item':copy.deepcopy(item)})

    def get_zfw_question_list(self,response):
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        next_url = re.findall("a> <a href=\"(.*?)\">下一页</a>",text)
        href_list = html.xpath("//li[@class='list-item']/div/a/@href")
        for href in href_list:
            yield scrapy.Request(url=href,callback=self.get_zfw_detail,meta={'item':copy.deepcopy(item)})
        if next_url:
            yield scrapy.Request(url=next_url[0],callback=self.get_zfw_question_list,meta={'item':copy.deepcopy(item)})

    def get_zfw_detail(self,response):
        item = response.meta['item']
        url = response.url
        item['url'] = url
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        question = html.xpath("//div[@class='container']/h1/text()")[0]
        item['question'] = question
        pubDate = html.xpath("//div[@class='container']/div[contains(@class,'wlinfo')]/div[1]/span[2]/text()")[0][:10]
        item['pubData'] = pubDate
        txt = html.xpath("//div[@class='article']//text()")
        answer = ''.join(txt)
        item['answer'] = answer
        item['model_type'] = "税务"
        item['uq_id'] = self.encode_md5(url+question+answer)
        yield item

    def flkc_parse(self,response):
        item = QuestItem()
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        a_list = html.xpath("//ul[@class='spe-ul']/li/a")
        next_page = re.findall("a> <a href=\"(.*?)\">下一页</a>",text)
        if next_page:
            next_href = "https://www.lawtime.cn/askzt/" + next_page[0]
            yield scrapy.Request(url=next_href,callback=self.flkc_parse)
        for a in a_list:
            href = a.xpath("./@href")[0]
            question_type = a.xpath("./@title")[0]
            item['question_type'] = question_type
            yield scrapy.Request(url=href,callback=self.get_flkc_quest_sign,meta={'item':copy.deepcopy(item)})
            print(f"{question_type}:{href}")

    def get_flkc_quest_sign(self,response):
        item = response.meta["item"]
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        next_page = re.findall("a> <a href=\"(.*?)\">下一页</a>", text)
        if next_page:
            next_href = "https://www.lawtime.cn/askzt/" + next_page[0]
            yield scrapy.Request(url=next_href,callback=self.get_flkc_quest_sign)
        a_list = html.xpath("//li[@class='item-li']/a")
        for a in a_list:
            question_sign = a.xpath("./@title")[0]
            href = a.xpath("./@href")[0]
            item['question_sign'] = question_sign
            yield scrapy.Request(url=href, callback=self.get_flkc_question_list,meta={'item':copy.deepcopy(item)})

    def get_flkc_question_list(self,response):
        item = response.meta["item"]
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        a_list = html.xpath("//div[@class='ans-swiper-card-item']/div[1]/a")
        for a in a_list:
            question = a.xpath("./text()")[0]
            item["question"] = question
            detail_url = a.xpath("./@href")[0]
            yield scrapy.Request(url=detail_url,callback=self.get_flkc_detail,meta={'item':copy.deepcopy(item)})

    def get_flkc_detail(self,response):
        print(f"flkc:{response.url}")
        item = response.meta['item']
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        ans_list = html.xpath("//div[@class='ans']")
        for ans in ans_list:
            answer = ans.xpath("./p[@class='ans-msg']//text()")
            if not answer:
                answer = ans.xpath("./div[@class='ans-msg']//text()")
            answer = "".join(answer)
            pubData = ans.xpath("./p[@class='ans-time']/text()")[0][:10]
            item['answer'] = answer
            item['pubData'] = pubData
            item['source'] = "法律快车"
            item['model_type'] = "税务"
            item["url"] = response.url
            item["uq_id"] = self.encode_md5(response.url+item["question"]+answer)
            yield item

    def hualv_parse(self,response):
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        max_page = html.xpath("//div[contains(@class,'u-page')]/a/text()")[-2]
        for index in range(1,int(max_page)+1):
            page_url = f"https://www.66law.cn/zhinan/qiyebanshi/qiyenashui/page_{index}.aspx"
            yield scrapy.Request(url=page_url,callback=self.get_hualv_quest_list)
            # break


    def get_hualv_quest_list(self,response):
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        # 企业纳税
        href_list = html.xpath("//ul[@class='tw-list']/li/h3/a/@href")
        for href in href_list:
            url = "https://www.66law.cn" + href
            # print(url)
            yield scrapy.Request(url=url,callback=self.get_hualv_detail)
            # break

    def get_hualv_detail(self,response):
        item = QuestItem()
        text = response.body.decode('utf-8')
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
            item['answer'] = re.sub("<.*?>","",content)
        # print(answer)
        # print("8"*100)
            url = response.url
            question_type = "企业纳税"
            item['question'] = question
            item['source'] = source
            item['pubData'] = pubDate
            # item['answer'] = answer
            item['url'] = url
            item['question_type'] = question_type
            item['model_type'] = "税务"
            item['uq_id'] = self.encode_md5(url+question+answer)
            yield item

