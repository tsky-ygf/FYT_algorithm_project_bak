#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/8 13:51 
@Desc    : None
"""
import logging
from typing import List, Dict
import pymysql


def get_news_by_id_list(id_list: List[int]) -> List[Dict]:
    # 打开数据库连接
    db = pymysql.connect(host='rm-bp1959t33moq9r35nco.mysql.rds.aliyuncs.com',
                         user='law',
                         password='Law220715',
                         database='law')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    try:
        format_strings = ','.join(['%s'] * len(id_list))
        # 执行SQL语句
        cursor.execute("SELECT id, title, release_time, content, raw_content, url FROM popular WHERE id in (%s) ORDER BY release_time DESC" % format_strings,
                       tuple(id_list))
        # 获取所有记录列表
        fetched_data = cursor.fetchall()
        news = [{"id": row[0], "title": row[1], "release_time": row[2], "content": row[3], "raw_content": row[4], "source_url": row[5]} for row in fetched_data]
    except:
        logging.error("Error: unable to fetch data")
        news = []
    # 关闭数据库连接
    db.close()
    return news
