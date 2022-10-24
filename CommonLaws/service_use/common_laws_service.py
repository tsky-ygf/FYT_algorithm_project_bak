#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/20 11:16
# @Author  : yuguofeng
# @Site    :
# @File    : consultation_service.py
# @Software: PyCharm
import re

import pymysql

common_laws_map = {
    "税法专栏":"swj_hot_news",
    "司法专栏":"sfj_hot_news",
    "金融专栏":"banking_hot_news",
    "市场监督":"scjd_hot_news",
    "法院专栏":"fy_hot_news",
    "公安专栏":"ga_hot_news",
    "文旅专栏":"wl_hot_news",
    "环保专栏":"hb_hot_news",
    "交通专栏":"jt_hot_news",
    "科技专栏":"kj_hot_news",
}
def get_table(category):
    table_name = common_laws_map.get(category)
    return table_name

def get_preview_data(tableName):
    select_sql = f"""select uq_id,title,pubDate,source,content from hot_news.{tableName} order by pubDate desc limit 30;"""
    conn = pymysql.connect(
        host="101.69.229.138",
        port=8501,
        user="root",
        password="Nblh@2022",
        db="hot_news",
        charset='utf8',
        cursorclass = pymysql.cursors.DictCursor
    )
    curs = conn.cursor()
    curs.execute(select_sql)
    res_list = curs.fetchall()
    curs.close()
    conn.close()
    preview_data_list = []
    if res_list:
        for data_item in res_list:
            content = data_item.pop('content')
            if content:
                content = re.sub("\s", "", content)
                data_item['preview'] = content[:20]
                data_item['tableName'] = tableName
                preview_data_list.append(data_item)
    return preview_data_list

def get_news(uq_id,tableName):
    select_sql = f"""select url,htmlContent,title,pubDate,source,content from hot_news.{tableName} where uq_id='{uq_id}';"""
    conn = pymysql.connect(
        host="101.69.229.138",
        port=8501,
        user="root",
        password="Nblh@2022",
        db="hot_news",
        charset='utf8',
        cursorclass = pymysql.cursors.DictCursor
    )
    curs = conn.cursor()
    curs.execute(select_sql)
    res_list = curs.fetchall()
    curs.close()
    conn.close()
    preview_data_list = []
    if res_list:
        for data_item in res_list:
            content = data_item.pop('content')
            if content:
                content = re.sub("\s", "", content)
                data_item['preview'] = content[:20]
                preview_data_list.append(data_item)
    if preview_data_list:
        return preview_data_list[0]
    else:
        return ''

if __name__ == '__main__':
    print(get_preview_data("swj_hot_news"))