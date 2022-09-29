#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 29/8/2022 15:25 
@Desc    : None
"""
import logging
from typing import Dict
import pymysql

from ProfessionalSearch.RelevantLaws.api.constants import (
    SEPERATOR_BETWEEN_LAW_TABLE_AND_ID,
)


def get_law_by_law_id(law_id: str, table_name: str) -> Dict:
    # 打开数据库连接
    db = pymysql.connect(
        host="172.19.82.227",
        user="root",
        password="Nblh@2022",
        database="falvfagui_data",
    )

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    try:
        # 执行SQL语句
        sql = "SELECT md5Clause, title, source, isValid, locality, resultChapter, resultSection, resultClause FROM {} WHERE md5Clause = '{}'".format(
            table_name, law_id
        )
        print(sql)
        cursor.execute(sql)
        # 获取所有记录列表
        fetched_data = cursor.fetchall()
        laws = [
            {
                "law_id": str(table_name)
                + SEPERATOR_BETWEEN_LAW_TABLE_AND_ID
                + str(row[0]),
                "law_name": row[1],
                "law_type": row[2],
                "timeliness": "现行有效" if str(row[3]) == "有效" else str(row[3]),
                "using_range": row[4] if row[4] else "全国",
                "law_chapter": row[5],
                "law_item": row[6],
                "law_content": row[7],
            }
            for row in fetched_data
        ]
    except:
        logging.error("Error: unable to fetch data")
        laws = []
    # 关闭数据库连接
    db.close()
    return laws[0] if laws else dict()


if __name__ == "__main__":
    table_name = "flfg_result_falv"
    law_id = "5a43120b27fe0457634a7420283b4aad"
    print(get_law_by_law_id(law_id, table_name))
