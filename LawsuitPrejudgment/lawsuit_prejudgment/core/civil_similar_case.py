#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 3/9/2022 12:12 
@Desc    : None
"""
import requests

id2table = {
    "6": "case_list_original_labor",
    "15": "case_list_original_labor_2",
    "16": "case_list_original_labor_2",
    "17": "case_list_original_hunyinjiating",
    "19": "case_list_original_hunyinjiating",
    "20": "case_list_original_qinquan",
    "21": "case_list_original_hetong",
    "22": "case_list_original_hetong",
    "23": "case_list_original_hetong",
    "24": "case_list_original_hetong",
    "25": "case_list_original_hetong",
    "26": "case_list_original_hetong",
    "27": "case_list_original_hetong",
    "28": "case_list_original_hetong",
    "29": "case_list_original_hetong",
    "30": "case_list_original_hetong",
    "31": "case_list_original_hetong",
    "32": "case_list_original_hetong",
    "33": "case_list_original_hetong",
    "34": "case_list_original_hetong",
    "35": "case_list_original_hetong",
    "36": "case_list_original_hetong",
    "37": "case_list_original_hetong",
    "38": "case_list_original_hetong",
    "39": "case_list_original_hetong",
    "40": "case_list_original_hetong",
    "41": "case_list_original_hetong",
    "42": "case_list_original_hetong",
    "43": "case_list_original_hetong",
    "44": "case_list_original_hetong",
    "45": "case_list_original_hetong",
    "46": "case_list_original_hetong",
    "47": "case_list_original_hetong",
    "48": "case_list_original_hetong",
    "49": "case_list_original_hetong",
    "50": "case_list_original_hetong",
    "51": "case_list_original_hetong",
    "52": "case_list_original_hetong",
    "53": "case_list_original_hetong",
    "54": "case_list_original_rengequan",
    "55": "case_list_original_rengequan",
    "56": "case_list_original_rengequan",
    "57": "case_list_original_rengequan",
    "58": "case_list_original_rengequan",
    "59": "case_list_original_rengequan",
    "60": "case_list_original_qinquan",
    "61": "case_list_original_qinquan",
    "62": "case_list_original_qinquan",
    "63": "case_list_original_qinquan",
    "64": "case_list_original_qinquan",
    "65": "case_list_original_qinquan",
    "66": "case_list_original_qinquan",
    "67": "case_list_original_qinquan",
    "68": "case_list_original_qinquan",
    "69": "case_list_original_qinquan",
    "70": "case_list_original_qinquan",
    "71": "case_list_original_qinquan",
    "72": "case_list_original_qinquan",
    "73": "case_list_original_qinquan",
    "74": "case_list_original_qinquan",
    "75": "case_list_original_qinquan",
    "76": "case_list_original_qinquan",
    "77": "case_list_original_wuquan",
    "78": "case_list_original_wuquan",
    "79": "case_list_original_wuquan",
    "80": "case_list_original_wuquan",
    "81": "case_list_original_wuquan",
    "82": "case_list_original_wuquan",
    "83": "case_list_original_wuquan",
    "84": "case_list_original_wuquan",
    "85": "case_list_original_wuquan",
    "86": "case_list_original_wuquan",
    "96": "case_list_original_labor",
    "97": "case_list_original_zhishichanquan",
    "98": "case_list_original_zhishichanquan",
    "99": "case_list_original_zhishichanquan",
    "100": "case_list_original_zhishichanquan"
}

import logging
from typing import List, Dict
import pymysql


def _get_civil_law_documents_by_id_list(id_list: List[str], table_name) -> List[Dict]:
    # 打开数据库连接
    db = pymysql.connect(host='172.19.82.153',
                         user='root',
                         password='123456',
                         database='justice_big_data')

    # 使用cursor()方法获取操作游标
    cursor = db.cursor()

    # SQL 查询语句
    try:
        format_strings = ','.join(['%s'] * len(id_list))
        # 执行SQL语句
        cursor.execute("SELECT f2, f1, f5, f14, f3, f41, f7 FROM " + table_name + " WHERE f2 in (%s)" % format_strings,
                       tuple(id_list))
        # 获取所有记录列表
        fetched_data = cursor.fetchall()
        law_documents = [{
            "doc_id": row[0],
            "doc_title": row[1],
            "case_number": row[2],
            "judge_date": row[3],
            "province": row[4],
            "court": row[5],
            "raw_content": row[6]
        } for row in fetched_data]
    except:
        logging.error("Error: unable to fetch data")
        law_documents = []
    # 关闭数据库连接
    db.close()
    return law_documents


def get_civil_law_documents_by_id_list(id_list: List[str], table_name=None) -> List[Dict]:
    if table_name:
        return _get_civil_law_documents_by_id_list(id_list, table_name)
    table_names = list(set(id2table.values()))
    for table_name in table_names:
        law_documents = _get_civil_law_documents_by_id_list(id_list, table_name)
        if law_documents:
            return law_documents
    return []


class CivilSimilarCase:
    def __init__(self, fact, problem, claim_list, problem_id):
        self.fact = fact
        self.problem = problem
        self.claim_list = claim_list
        self.problem_id = problem_id
        self.table_name = id2table.get(str(self.problem_id), "case_list_original_hetong")

    def get_similar_cases(self):
        url = "http://172.19.82.198:5011/top_k_similar_narrative"
        body = {
            "fact": str(self.fact) + " " + self.problem + " " + "".join(self.claim_list),
            "problem": self.problem,
            "claim_list": self.claim_list
        }
        resp_json = requests.post(url, json=body).json()
        return self._get_short_document(resp_json)

    def _get_short_document(self, resp_json, top_k=10):
        doc_ids = resp_json["dids"][:top_k]
        sim_list = resp_json["sims"][:top_k]
        tags_list = resp_json["tags"][:top_k]
        law_documents = get_civil_law_documents_by_id_list(doc_ids, self.table_name)
        if not law_documents:
            return []
        return [
            {
                "doc_id": item["doc_id"],
                "similar_rate": next((sim for idx, sim in enumerate(sim_list) if str(doc_ids[idx]) == str(item["doc_id"])), 0.6),
                "title": item["doc_title"],
                "court": item["court"],
                "judge_date": item["judge_date"],
                "case_number": item["case_number"],
                "tag": next((tag for idx, tag in enumerate(tags_list) if str(doc_ids[idx]) == str(item["doc_id"])), ""),
                "is_guiding_case": False
            }
            for item in law_documents
        ]
