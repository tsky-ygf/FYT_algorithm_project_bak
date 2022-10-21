import hashlib
import re
import scrapy
from wl_hot_news.items import WlHotNewsItem
from lxml import etree
import redis

class ZglybSpider(scrapy.Spider):
    name = 'zglyb'
    start_urls = ['https://www.mct.gov.cn/whzx/zxgz/']
    redis_key = "wl_hot_news"
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)

    def parse(self, response):
        text = response.body.decode('utf-8')
        whyw_sign = re.findall("<a href=\"\.\.(.*?)\" target=\"_blank\" title=\"焦点新闻\">焦点新闻</a>",text)[0]
        ldhd_sign = re.findall("<a href=\"\.\.(.*?)\" target=\"_blank\" title=\"领导活动\">领导活动</a>",text)[0]
        whyw_url = "https://www.mct.gov.cn/whzx" + whyw_sign
        ldhd_url = "https://www.mct.gov.cn/whzx" + ldhd_sign
        yield scrapy.Request(url=whyw_url, dont_filter=True, callback=self.get_page)
        yield scrapy.Request(url=ldhd_url, dont_filter=True, callback=self.get_page)
        yield scrapy.Request(url=whyw_url, dont_filter=True, callback=self.get_detail_list)
        yield scrapy.Request(url=ldhd_url, dont_filter=True, callback=self.get_detail_list)
    def get_page(self, response):
        text = response.body.decode('utf-8')
        max_page = re.findall("createPageHTML\((.*?), .*?, \"index\", \"htm\"\);",text)[0]
        for index in range(1,int(max_page)):
            url = response.url + f"index_{index}.htm"
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_list)

    def get_detail_list(self, response):
        resp_url = response.url
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        tr_list = html.xpath("//div[@class='bt-rig-cen-01']/table/tr")

        for tr in tr_list:
            href = tr.xpath("./td/a/@href")[0]
            if "whyw" in resp_url:
                url = get_whyw_url(href)

            elif 'ldhd' in resp_url:
                url = get_whyw_url(href)
            else:
                url = ''
            if url and self.redis_conn.sadd(self.redis_key,url):
                yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail)
            # yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail)
            # break

    def get_detail(self, response):
        item = WlHotNewsItem()
        resp_url = response.url
        text = response.body.decode('utf-8')
        html = etree.HTML(text)
        trs = html.xpath("//div[@class='TRS_Editor']")[0]
        htmlContent = etree.tostring(trs,encoding='utf-8',pretty_print=True, method='html').decode('utf-8')
        front_url = re.findall("(.*\/)", resp_url)[0]
        htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
        if "whyw" in resp_url:
            htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
            question_type = "焦点新闻"
        elif 'ldhd' in resp_url:
            htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
            question_type = "领导活动"
        else:
            question_type = ''

        title = html.xpath("//meta[@name='ArticleTitle']/@content")[0]
        pubDate = html.xpath("//meta[@name='PubDate']/@content")[0][:10]
        content = "".join(trs.xpath(".//text()"))
        content = re.sub("\.TRS.*;}", "", content)
        item = {
            'url':resp_url,
            'uq_id':self.encode_md5(resp_url),
            'title':title,
            'content':content,
            'htmlContent':htmlContent,
            'source':"中国文化旅游部",
            'pubDate':pubDate,
            'category':"文旅专栏",
            'question_type':question_type,
        }
        yield item


    def encode_md5(self,value):
        if value:
            hl = hashlib.md5()
            hl.update(value.encode(encoding='utf-8'))
            return hl.hexdigest()
        return ''
def get_whyw_url(url):
    if re.match("^\./",url):
        return url.replace("./","https://www.mct.gov.cn/whzx/whyw/")
    else:
        return ''
    pass

def get_ldhd_url(url):
    if re.match("^http", url):
        return url
    else:
        return ''

# if __name__ == '__main__':
#     text = """
# .TRS_Editor P{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor DIV{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor TD{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor TH{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor SPAN{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor FONT{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor UL{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor LI{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}.TRS_Editor A{margin-top:5px;margin-bottom:5px;line-height:1.5;font-family:宋体;font-size:12pt;}
# 　　9月30日，文化和旅游部党组理论学习中心组专题学习《习近平谈治国理政》第四卷，结合工作实际交流研讨。文化和旅游部党组书记、部长胡和平主持会议并讲话，部党组成员李群、王旭东、腊翊凡、卢映川、杜江、饶权出席会议并发言，文化和旅游部党的二十大代表参加学习。
# 　　大家在学习研讨中表示，《习近平谈治国理政》第四卷与前三卷浑然一体、相得益彰，是全面完整系统记载和反映习近平新时代中国特色社会主义思想成果的重要文献。这部重要著作，着眼学思践悟笃行、遵循理论创新逻辑，着眼增强历史自信、遵循历史主动逻辑，着眼记录伟大历程、遵循实践发展逻辑，着眼提供中国方案、遵循国际传播逻辑，对于广大党员干部深入学习贯彻习近平新时代中国特色社会主义思想，深刻领悟“两个确立”的决定性意义，增强“四个意识”、坚定“四个自信”、做到“两个维护”，具有重要意义；对于帮助海外读者、国际社会及时了解、正确认识这一伟大思想的最新发展成果，为党和国家事业发展创造良好外部环境，具有重要意义。
# 　　大家认为，《习近平谈治国理政》第四卷彰显了习近平新时代中国特色社会主义思想的体系性创新，全面系统体现了从“八个明确”到“十个明确”的丰富发展，共同构成这一伟大思想的“四梁八柱”。要深入学习领会这部重要著作的丰富内涵、深邃思想、精神实质、实践要求，深刻领悟“两个确立”的决定性意义，深刻认识“两个结合”的理论品质，深刻理解党的百年奋斗历史经验的科学内涵和精神要义。要深刻把握社会主义文化强国建设的目标任务，系统总结新时代文化建设的伟大成就和重要经验，充分认识社会主义文化强国的特质，坚定不移走中国特色社会主义文化发展道路，不断铸就中华文化新辉煌。要深刻领会全面从严治党、坚持自我革命的重要要求，保持越是艰险越向前的斗争精神，保持全面从严治党永远在路上的坚定清醒，确保党始终成为中国特色社会主义事业、推动文化和旅游高质量发展的坚强领导核心。
# 　　会议要求，要把学习贯彻《习近平谈治国理政》第四卷作为当前和今后一个时期的一项重要政治任务，切实增强政治自觉、思想自觉和行动自觉。要提高政治站位，坚持读原著学原文、悟原理知原义，做到学思用贯通、知信行统一。要周密组织实施，把学习宣传贯彻工作摆到重要位置，作为持续深化政治机关意识教育、对党忠诚教育的重要载体。要紧密结合实际，务求学习实效，推动习近平新时代中国特色社会主义思想在文化和旅游领域更好地落地生根、开花结果。党的二十大召开在即，文化和旅游部出席代表要加强履职学习，不断提高政治判断力、政治领悟力、政治执行力，增强履行好代表职责的思想自觉和政治自觉，为大会圆满成功作出应有贡献。
# 　　文化和旅游部各司局党政主要负责同志、驻部纪检监察组负责同志参加会议，文化和旅游部党的二十大代表以及公共服务司、国际交流与合作局负责同志作交流发言。"""
#     print(re.sub("\.TRS.*;}","",text))
#     url = """<div style="text-align: center;"><font style="line-height: 150%;"><img src="./W020220929395307167269.jpg" width="600" style="border-width: 0px;" """
#     res_url = url.replace("img src=\"./","img src=\"https://www.mct.gov.cn/whzx/whyw/202209/")
#     print(res_url)
