# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymysql
from pymysql.converters import escape_string

class SwjHotNewsPipeline:

    def __init__(self):
        self.connect()

    def connect(self):
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
        # 创建游标
        self.cursor = self.conn.cursor()

    def process_item(self, item, spider):
        # s = "uq_id,title,content,province,pubDate,url"
        temp_item = self.escape_dict(item)
        insert_sql = """insert into hot_news.swj_hot_news (uq_id,title,content,province,pubDate,url,source,category,htmlContent) 
                values ('{}','{}','{}','{}','{}','{}','{}','{}','{}')""".format(temp_item.get('uq_id'), temp_item.get('title'),
                                                                      temp_item.get('content'),
                                                                      temp_item.get('province'),
                                                                      temp_item.get('pubDate'), temp_item.get('url'),
                                                                      temp_item.get('source'), temp_item.get('category'),
                                                                      temp_item.get('htmlContent'))
        insert_sql = insert_sql.replace("\'None\'","\'\'")
        self.conn.ping(reconnect=True)
        try:
            self.cursor.execute(insert_sql)
            self.conn.commit()
        except pymysql.err.IntegrityError:
            print(f"主键'{temp_item.get('uq_id')}重复'")
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
