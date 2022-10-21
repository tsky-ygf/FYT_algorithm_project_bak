import json
import re
import time
import scrapy
from lxml import etree

class XfDataSpider(scrapy.Spider):
    name = 'xf_data'
    start_urls = ['https://flk.npc.gov.cn/api/']

    def start_requests(self):
        for url in self.start_urls:
            time_sign = int(time.time() * 1000)
            yield scrapy.Request(url=f"https://flk.npc.gov.cn/api/?page=1&type=xffl&searchType=title%3Baccurate&sortTr=f_bbrq_s%3Bdesc&gbrqStart=&gbrqEnd=&sxrqStart=&sxrqEnd=&size=10&_={time_sign}",callback=self.parse)

    def parse(self, response):
        text = response.body.decode("utf-8")
        print(text)
        # html = etree.HTML(text)
        # li_list = html.xpath("//li[@class='l-wen']")
        # for li in li_list:
        #     title = li.xpath("./@title")[0]
        #     # onclick_text = etree.tostring(onclick,encoding='utf-8', method='html', pretty_print=True).decode('utf-8')
        #     onclick_text = li.xpath("./@onclick")[0]
        #     # print(title, onclick_text)
        #     href = re.findall("showDetail\(\"\.(.*?)\"\)", onclick_text)[0]
        #     print(title,href)

# if __name__ == '__main__':
#     import time
#     time_sign = int(time.time()*1000)
#     print(time_sign)

    # 1666169079.7667992
    # 1666168699491
    # 1666169148554