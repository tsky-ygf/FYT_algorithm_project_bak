import scrapy
import time
from loguru import logger
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
# 关闭ssl验证提示
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import re
import execjs
import hashlib
import json

def getCookie(data):
    """
    通过加密对比得到正确cookie参数
    :param data: 参数
    :return: 返回正确cookie参数
    """
    chars = len(data['chars'])
    for i in range(chars):
        for j in range(chars):
            clearance = data['bts'][0] + data['chars'][i] + data['chars'][j] + data['bts'][1]
            encrypt = None
            if data['ha'] == 'md5':
                encrypt = hashlib.md5()
            elif data['ha'] == 'sha1':
                encrypt = hashlib.sha1()
            elif data['ha'] == 'sha256':
                encrypt = hashlib.sha256()
            encrypt.update(clearance.encode())
            result = encrypt.hexdigest()
            if result == data['ct']:
                return clearance

class ZggabSpider(scrapy.Spider):
    name = 'zggab'
    start_urls = ['https://www.mps.gov.cn/n2254098/n4904352/index.html']

    def __init__(self):
        self.headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Referer": "https://www.mps.gov.cn/n2254098/n4904352/index.html",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36",
            "sec-ch-ua": "\"Google Chrome\";v=\"105\", \"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"105\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\""
        }
        self.cookies = {
            "maxPageNum7627565": "857",
            # "__jsluid_s": "9ec7c374a201f5f42d9eb601ef8dd5c9",
            # "__jsluid_s": "9ec7c374a201f5f42d9eb601ef8dd5c9",
            # "__jsl_clearance_s": "1664502886.442|0|2snpITh1%2BxE7cZbv10hgUZCVL18%3D"
        }

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, cookies=self.cookies, dont_filter=True, callback=self.gab_parse)
    # start_urls = ['https://www.baidu.com']

    def gab_parse(self, response, **kwargs):
        url = response.url
        print(url)
        status = response.status
        if status == 521:
            self.set_cookies()
            yield scrapy.Request(url=url, cookies=self.cookies, dont_filter=True, callback=self.gab_parse)
        elif status == 200:
            print(response.body.decode('utf-8'))

    def get_response_text(self):
        url = "https://www.mps.gov.cn/n2253534/n2253535/index.html"
        response = requests.get(url=url,headers=self.headers, cookies=self.cookies)
        logger.info(f"get_response_text old cookies:{response.cookies.get_dict()}")
        if response.cookies.get_dict().get('__jsluid_s'):
            self.cookies['__jsluid_s'] = response.cookies.get_dict().get('__jsluid_s')
            logger.info(f"get_response_text cookies:{self.cookies}")
        text = response.text
        return text

    def get_jsl_clearance(self):
        text = self.get_response_text()
        logger.info(f"frist request:{text}")
        resp_js = re.findall('cookie=(.*?);location.href=', text)[0]
        jsl_clearance = execjs.eval(resp_js)
        jsl_clearance = re.findall("__jsl_clearance_s=(.*?);", jsl_clearance)[0]
        logger.info(f"__jsl_clearance: {jsl_clearance}")
        self.cookies['__jsl_clearance_s'] = jsl_clearance

    def set_cookies(self):
        self.get_jsl_clearance()
        time.sleep(3)
        text = self.get_response_text()
        logger.info(text)
        data = json.loads(re.findall(r';go\((.*?)\)', re.sub("\s", "", text))[0])
        jsl_clearance_s = getCookie(data)
        logger.info(f"second jsl_clearance_s: {jsl_clearance_s}")
        self.cookies['__jsl_clearance_s'] = jsl_clearance_s
# if __name__ == '__main__':
#     GetCookies().get_cookies()
