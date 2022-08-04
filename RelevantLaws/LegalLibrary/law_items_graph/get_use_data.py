#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/1 14:59
# @Author  : Adolf
# @Site    : 
# @File    : get_use_data.py
# @Software: PyCharm
import pandas as pd
import pymysql

# 连接数据库
# 加上charset='utf8'，避免 'latin-1' encoding 报错等问题
conn = pymysql.connect(host='172.19.82.227',
                       user='root',
                       password='Nblh@2022',
                       db='judgments_data',
                       charset='utf8')

# 创建cursor
cursor = conn.cursor()
sql = "SELECT fl_ft FROM judgment_minshi_data where fl_ft is not null and event_type='判决书' limit 1000;"

# 执行sql语句
cursor.execute(sql)
# 获取数据库列表信息
# col = cursor.description

# 获取全部查询信息
re = cursor.fetchall()
# 获取一行信息
# re = cursor.fetchone()

# 获取的信息默认为tuple类型，将columns转换成DataFrame类型
# columns = pd.DataFrame(list(col))
# 将数据转换成DataFrame类型，并匹配columns
# df = pd.DataFrame(list(re), columns=columns[0])
df = pd.DataFrame(list(re), columns=['law_items'])
print(df)
df.to_csv("data/law/law_lib/item_test.csv", index=False)
