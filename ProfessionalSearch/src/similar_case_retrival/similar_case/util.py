import logging
from typing import List, Dict
import pymysql

def get_civil_law_documents_by_id_list(id_list: List[str], table_name) -> List[Dict]:
    # 打开数据库连接
    db = pymysql.connect(host='172.19.82.227',
                         user='root',
                         password='Nblh@2022',
                         database='judgments_data')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    try:
        format_strings = ','.join(['%s'] * len(id_list))
        # 执行SQL语句
        cursor.execute("SELECT uq_id, jslcm FROM " + table_name + " WHERE uq_id in (%s)" % format_strings,
                       tuple(id_list))
        # 获取所有记录列表
        fetched_data = cursor.fetchall()
        law_documents = [{
            "uq_id": row[0],
            "jslcm": row[1]
        } for row in fetched_data]
    except:
        logging.error("Error: unable to fetch data")
        law_documents = []
    # 关闭数据库连接
    db.close()
    return law_documents