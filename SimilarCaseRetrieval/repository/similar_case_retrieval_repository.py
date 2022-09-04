#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 23/8/2022 15:05 
@Desc    : None
"""
import logging
from typing import List, Dict
import pymysql


# def get_criminal_law_documents_by_id_list(id_list: List[str]) -> List[Dict]:
#     # 打开数据库连接
#     db = pymysql.connect(host='172.19.82.153',
#                          user='root',
#                          password='123456',
#                          database='justice_big_data')
#
#     # 使用cursor()方法获取操作游标
#     cursor = db.cursor()
#
#     # SQL 查询语句
#     try:
#         format_strings = ','.join(['%s'] * len(id_list))
#         # 执行SQL语句
#         cursor.execute("SELECT f8, f28, f3, f29, f5, f10, f30, f31, f32, f33, f34, f35, f36 FROM case_new_criminal WHERE f8 in (%s)" % format_strings,
#                        tuple(id_list))
#         # 获取所有记录列表
#         fetched_data = cursor.fetchall()
#         law_documents = [{
#             "doc_id": row[0],
#             "doc_title": row[1],
#             "case_number": row[2],
#             "judge_date": row[3],
#             "province": row[4],
#             "court": row[5],
#             "text_header": row[6],
#             "party_information": row[7],
#             "litigation_record": row[8],
#             "basic_information": row[9],
#             "judging_basis": row[10],
#             "judging_result": row[11],
#             "end_of_text": row[12]
#         } for row in fetched_data]
#     except:
#         logging.error("Error: unable to fetch data")
#         law_documents = []
#     # 关闭数据库连接
#     db.close()
#     return law_documents


def get_criminal_law_documents_by_id_list(id_list: List[str]) -> List[Dict]:
    # 打开数据库连接
    db = pymysql.connect(host='rm-8vbhaq1jlv38l7qeyoo.mysql.zhangbei.rds.aliyuncs.com',
                         user='baoxiaohei_python',
                         password='Baoxiaohei@2022',
                         database='judgments_data')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    try:
        format_strings = ','.join(['%s'] * len(id_list))
        # 执行SQL语句
        cursor.execute("SELECT uq_id, htmlContent FROM judgment_xingshi_data WHERE uq_id in (%s)" % format_strings,
                       tuple(id_list))
        # 获取所有记录列表
        fetched_data = cursor.fetchall()
        law_documents = [{
            "doc_id": row[0],
            "html_content": row[1]
        } for row in fetched_data]
    except:
        logging.error("Error: unable to fetch data")
        law_documents = []
    # 关闭数据库连接
    db.close()
    return law_documents