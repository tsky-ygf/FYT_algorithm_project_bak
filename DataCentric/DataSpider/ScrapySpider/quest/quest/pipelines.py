# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymysql
from pymysql.converters import escape_string


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

    def process_item(self, item, spider):
        temp_item = self.escape_dict(item)
        insert_sql = """insert into quest_answer_data.quest_answer_table (uq_id,question_type,question_sign,question,pubData,hot_question_sign,answer,source,model_type,url) 
        values ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')""".format(temp_item.get('uq_id'),temp_item.get('question_type'),temp_item.get('question_sign'),temp_item.get('question'),temp_item.get('pubData'),temp_item.get('hot_question_sign'),temp_item.get('answer'),temp_item.get('source'),temp_item.get('model_type'),temp_item.get('url'))
        insert_sql = insert_sql.replace("\'None\'","\'\'")
        print(insert_sql)
        self.conn.ping(reconnect=True)
        self.cursor.execute(insert_sql)
        self.conn.commit()
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
