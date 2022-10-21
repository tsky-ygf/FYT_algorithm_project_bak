# coding:utf-8

# @Time : 2022/9/30 10:24

# @Author : yuGuoFeng

# @File : gab_zhrmghg.py

# @Software: PyCharm
# _*_ coding: utf-8 _*_
# @Time : 2022/3/9 3:54 下午
# @Author : 于国峰
# @File : tax_error.py
# @Describe:
import json
from hashlib import md5
import time

import redis
from loguru import logger
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
# 关闭ssl验证提示
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import re
import execjs
import hashlib
import json
import pymysql
from pymysql.converters import escape_string
from lxml import etree

class GetCookies():

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
            "__jsluid_s": "9ec7c374a201f5f42d9eb601ef8dd5c9",
            "__jsl_clearance_s": "1664582757.505|0|l%2FyAujSEXZZU9jvsINLoh9sjnw8%3D"
        }
        self.mysql = GabHotNewsPipeline()
        self.reload_cookies = 1
        self.data_dict = {}
        self.redis_conn = redis.Redis(host="124.221.199.74", port=6001)
        self.redis_key = "ga_hot_news"

    def get_response_text(self):
        url = "https://www.mps.gov.cn/n2253534/n2253535/index.html"
        response = requests.get(url=url,headers=self.headers, cookies=self.cookies)
        if response.cookies.get_dict().get('__jsluid_s'):
            self.cookies['__jsluid_s'] = response.cookies.get_dict().get('__jsluid_s')
        text = response.text
        return text

    def get_jsl_clearance(self):
        text = self.get_response_text()
        resp_js = re.findall('cookie=(.*?);location.href=', text)[0]
        jsl_clearance = execjs.eval(resp_js)
        jsl_clearance = re.findall("__jsl_clearance_s=(.*?);", jsl_clearance)[0]
        self.cookies['__jsl_clearance_s'] = jsl_clearance

    def set_cookies(self):
        logger.info(f"set new cookies")
        self.get_jsl_clearance()
        time.sleep(2)
        text = self.get_response_text()
        data = json.loads(re.findall(r';go\((.*?)\)', re.sub("\s", "", text))[0])
        jsl_clearance_s = getCookie(data)
        self.reload_cookies += 1
        self.cookies['__jsl_clearance_s'] = jsl_clearance_s

    def get_parse(self):
        url = "https://www.mps.gov.cn/n2253534/n2253535/index.html"
        # url = "https://www.mps.gov.cn/n2253534/n4904351/index_7574611.html"
        response = requests.get(url=url, headers=self.headers, cookies=self.cookies)
        if response.status_code == 521:
            if self.reload_cookies > 20:
                return
            self.set_cookies()
            self.get_parse()
            return
        text = response.content.decode('utf-8')
        html = etree.HTML(text)
        a_list = html.xpath("//ul[@class='listTitle']/a")
        for index,item in enumerate(a_list):
            if index in [1,3,4,7]:
                question_type = item.xpath("./li/font/text()")[0]
                href = item.xpath("./@href")[0]
                url = href.replace("../../","https://www.mps.gov.cn/")
                # logger.info(url)
                self.data_dict = {
                    "question_type": question_type,
                    "category": "公安专栏",
                    "source": "中华人民共和国公安部",
                }
                self.get_all_page_url(url)
                time.sleep(2)
                # break

    def get_all_page_url(self,url):
        # url = https://www.mps.gov.cn/n2253534/n2253535/index.html
        response = requests.get(url=url, headers=self.headers, cookies=self.cookies)
        if response.status_code == 521:
            if self.reload_cookies > 20:
                return
            self.set_cookies()
            self.get_all_page_url(url)
            return
        text = response.content.decode('utf-8')
        html = etree.HTML(text)
        li_list = html.xpath("//ul[@class='list']/li")
        for li in li_list:
            href = li.xpath("./a/@href")[0]
            if "http" in href:
                continue
            pubDate = li.xpath("./span/text()")[0]
            title = li.xpath("./a/text()")[0]
            detail_url = href.replace("../../","https://www.mps.gov.cn/")
            self.data_dict['url'] = detail_url
            self.data_dict['pubDate'] = pubDate
            self.data_dict['title'] = title
            self.get_detail(detail_url)
            # break
            time.sleep(1)
        page_list = re.findall("document.cookie=\"maxPageNum(.*?)=(.*?)\"",text)
        page_sign = int(page_list[0][0])
        page_num = int(page_list[0][1])
        # url = "https://www.mps.gov.cn/n2253534/n2253535/index.html"
        # url = "https://www.mps.gov.cn/n2253534/n2253535/index_7627565_856.html"
        for i in range(1,page_num)[::-1]:
            new_url = url.replace(f"index.html",f"index_{page_sign}_{i}.html")
            # logger.info(new_url)
            self.get_detail_url(new_url)
            time.sleep(1)
            # break

    def get_detail_url(self, url):

        time.sleep(2)
        response = requests.get(url=url, headers=self.headers, cookies=self.cookies)
        if response.status_code == 521:
            if self.reload_cookies > 20:
                return
            self.set_cookies()
            self.get_detail_url(url)
            return
        text = response.content.decode('utf-8')
        # logger.info(text)
        html = etree.HTML(text)
        li_list = html.xpath("//ul[@class='list']/li")
        for li in li_list:
            href = li.xpath("./a/@href")[0]
            if "http" in href:
                continue
            pubDate = li.xpath("./span/text()")[0]
            title = li.xpath("./a/text()")[0]
            # logger.info(href)
            detail_url = href.replace("../../", "https://www.mps.gov.cn/")
            self.data_dict['url'] = detail_url
            self.data_dict['title'] = title
            self.data_dict['pubDate'] = pubDate
            # logger.info(title)
            # logger.info(pubDate)
            # logger.info(detail_url)
            self.get_detail(detail_url)
            time.sleep(1)
            # break

    def get_detail(self, url):
        try:
            logger.info(f"get_detail method start, url: {url}")
            time.sleep(2)
            res = self.redis_conn.sadd(self.redis_key,url)
            if res == 0:
                return
            response = requests.get(url=url, headers=self.headers, cookies=self.cookies)
            if response.status_code == 521:
                if self.reload_cookies > 20:
                    return
                self.set_cookies()
                self.get_detail(url)
                return
            text = response.content.decode('utf-8')
            html = etree.HTML(text)
            div_content = html.xpath("//div[contains(@class,'wordContent')]")[0]
            htmlContent = etree.tostring(div_content,encoding='utf-8',method='html',pretty_print=True).decode('utf-8')
            content = "\n".join(re.findall("<p.*?2em.*?>(.*?)</p>", htmlContent))
            replace_str_list = re.findall('<img.*?src=\".*?\.jpg', htmlContent)
            replace_set = set()
            for replace_str in replace_str_list:
                replace_set.add(replace_str)
            if replace_set:
                for replace_item in replace_set:
                    if replace_item:
                        res_str = replace_item.replace("../../../", "https://www.mps.gov.cn/")
                        htmlContent = htmlContent.replace(replace_item, res_str)
            if content:
                self.data_dict['uq_id'] = encode_md5(url)
                self.data_dict['content'] = re.sub("<.*?>","",content)
                self.data_dict['htmlContent'] = htmlContent
                self.mysql.process_item(item=self.data_dict)
        except Exception as e:
            logger.error(f"get_detail method error: {e}")
            return

def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

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

def escape_dict(data_dict):
    for k, v in data_dict.items():
        if v:
            data_dict[k] = escape_string(v)
    return data_dict

class GabHotNewsPipeline:

    def __init__(self):
        self.conn = pymysql.connect(
            # host=self.host,
            host="101.69.229.138",
            # port=self.port,
            port=8501,
            user="root",
            password="Nblh@2022",
            db="hot_news",
            charset='utf8'
        )
        self.cursor = self.conn.cursor()

    def process_item(self, item):
        temp_item = escape_dict(item)
        insert_sql = """insert into hot_news.ga_hot_news (uq_id,title,content,province,pubDate,url,htmlContent,source,question_type,category) 
                values ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')""".format(temp_item.get('uq_id'),
                                                                                     temp_item.get('title'),
                                                                                     temp_item.get('content'),
                                                                                     temp_item.get('province'),
                                                                                     temp_item.get('pubDate'),
                                                                                     temp_item.get('url'),
                                                                                     temp_item.get('htmlContent'),
                                                                                     temp_item.get('source'),
                                                                                     temp_item.get('question_type'),
                                                                                     temp_item.get('category'))
        insert_sql = insert_sql.replace("\'None\'", "\'\'")
        # print(insert_sql)
        self.conn.ping(reconnect=True)
        try:
            logger.info(f"mysql insert start")
            self.cursor.execute(insert_sql)
            self.conn.commit()
            logger.info(f"insert into table ga_hot_news success, uq_id: {temp_item.get('uq_id')}")
        except pymysql.err.IntegrityError:
            logger.error(f"主键'{temp_item.get('uq_id')}重复'")
        except Exception as e:
            logger.error(e)

    def __del__(self):
        # 关闭游标
        self.cursor.close()
        # 关闭连接
        self.conn.close()

if __name__ == '__main__':
    GetCookies().get_parse()
    # s = '<img border="0" src="../../../n2253534/n4904351/c8675886/part/8675898.jpg">'
    # # res = re.sub('src="(.*?).*?.jpg"','https://www.mps.gov.cn/',s)
    # # res = re.findall('src=\"\.\/\.\.\/\.\.\/.*?\.jpg',s)
    # replace_str_list = re.findall('<img.*?src=\".*?\.jpg',s)
    # replace_set = set()
    # for replace_str in replace_str_list:
    #     replace_set.add(replace_str)
    # if replace_set:
    #     for replace_item in replace_set:
    #         if replace_item:
    #             res_str = replace_item.replace("../../../","https://www.mps.gov.cn/")
    #             print(s.replace(replace_item,res_str))

