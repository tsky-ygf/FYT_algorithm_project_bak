import hashlib

import requests
import re
from lxml import etree
import time
import pymysql
from pymysql.converters import escape_string

headers = {
    "authority": "www.66law.cn",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "referer": "https://www.66law.cn/laws/",
    "sec-ch-ua": "^\\^Chromium^^;v=^\\^106^^, ^\\^Google",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "^\\^Windows^^",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36"
}

def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

def test1():

    for i in range(1,41):
        url = f"https://www.66law.cn/laws/hetongfa/hetongjiufen/lvyoujf/page_{i}.aspx"
        test2(url)


def test2(url):
    time.sleep(3)
    response = requests.get(url, headers=headers)
    text = response.text
    html = etree.HTML(text)
    li_list = html.xpath("//ul[contains(@class,'cx-tw-list')]/li")
    for li in li_list:
        href = li.xpath("./a/@href")[0]
        url = "https://www.66law.cn" + href
        test3(url)

def test3(url):
    time.sleep(3)
    try:
        response = requests.get(url, headers=headers)
        text = response.text
        html = etree.HTML(text)
        pubDate = re.findall("<meta property=\"og:release_date\" content=\"(.*?)\" />",text)[0]
        question = re.findall("<meta name=\"keywords\" content=\"(.*?)\" />",text)[0]
        div = html.xpath("//div[@class='det-nr']")[0]
        htmlContent = etree.tostring(div, encoding='utf-8', pretty_print=True, method='html').decode('utf-8')
        # content = "".join(div.xpath(".//text()"))
        content = "\n".join(re.findall("<p.*?2em.*?>(.*?)</p>", htmlContent))
        content = re.sub("<.*?>","",content)
        txt_list = re.split("。",content)
        sub_index = []
        for index,txt_item in enumerate(txt_list):
            if txt_item and "华律网" in txt_item:
                sub_index.append(index)
        if sub_index:
            for index in sub_index:
                txt_list.pop(index)
        content = "。".join(txt_list)
        item = {
            "uq_id":encode_md5(url),
            "question_type":"旅游合同纠纷",
            "question":question,
            "pubData":pubDate,
            "answer":content,
            "source":"华律网",
            "model_type":"文旅专栏",
            "url":url,
        }
        pipeline.process_item(item)
    except Exception as e:
        return

class QuestPipeline:
    def __init__(self):
        self.host = "172.19.82.227"
        self.port = 3306
        self.user = "root"
        self.passwd = "Nblh@2022"
        self.db = "quest_answer_data"
        self.character = 'utf8'
        self.connect()

    def connect(self):
        self.conn = pymysql.connect(
            # host=self.host,
            host="101.69.229.138",
            # port=self.port,
            port=8501,
            user=self.user,
            password=self.passwd,
            db=self.db,
            charset=self.character
        )
        # 创建游标
        self.cursor = self.conn.cursor()

    def process_item(self, item):
        temp_item = self.escape_dict(item)
        insert_sql = """insert into quest_answer_data.quest_answer_table (uq_id,question_type,question_sign,question,pubData,hot_question_sign,answer,source,model_type,url) 
        values ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')""".format(temp_item.get('uq_id'),temp_item.get('question_type'),temp_item.get('question_sign'),temp_item.get('question'),temp_item.get('pubData'),temp_item.get('hot_question_sign'),temp_item.get('answer'),temp_item.get('source'),temp_item.get('model_type'),temp_item.get('url'))
        insert_sql = insert_sql.replace("\'None\'","\'\'")
        print(insert_sql)
        self.conn.ping(reconnect=True)
        try:
            self.cursor.execute(insert_sql)
            self.conn.commit()
        except Exception as e:
            print(e)
        return item

    def escape_dict(self, data_dict):
        for k, v in data_dict.items():
            if v:
                data_dict[k] = escape_string(v)
        return data_dict

    def __del__(self):
        # 关闭游标
        self.cursor.close()
        # 关闭连接
        self.conn.close()



if __name__ == '__main__':
    pipeline = QuestPipeline()
    test1()
