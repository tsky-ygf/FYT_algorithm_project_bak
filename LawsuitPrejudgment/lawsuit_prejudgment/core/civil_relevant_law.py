#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 3/9/2022 12:13 
@Desc    : None
"""
import requests
from pypinyin import lazy_pinyin


class CivilRelevantLaw:
    def __init__(self, fact, problem, claim_list):
        self.fact = fact
        self.problem = problem
        self.claim_list = claim_list

    @staticmethod
    def get_law_id(law_name, law_item):
        return "MEMORY_" + "".join(
            lazy_pinyin(str(law_name).replace("》", "").replace("《", "").strip())) + "-" + "".join(
            lazy_pinyin(str(law_item).strip()))

    @staticmethod
    def get_law_in_memory(law_id):
        try:
            url = "http://172.19.82.199:5090/get_law_in_memory"
            body = {
                "law_id": law_id
            }
            resp_json = requests.post(url, json=body).json()
            return resp_json.get("result")
        except Exception:
            return None

    @staticmethod
    def store_law_in_memory(law):
        try:
            url = "http://172.19.82.199:5090/store_law_in_memroy"
            resp_json = requests.post(url, json=law).json()
            return resp_json.get("success")
        except Exception:
            return None

    def _reformat(self, resp_json):
        law_name_and_items = resp_json.get("law_name_and_items")
        if not law_name_and_items:
            return []

        law_name_and_items.sort(key=lambda x: x[3], reverse=True)
        law_name_and_items = law_name_and_items[:10]
        result = []
        for item in law_name_and_items:
            law_id = self.get_law_id(item[0], item[1])
            content = {
                "law_id": law_id,
                "law_name": item[0],
                "law_item": item[1],
                "law_content": str(item[1]) + ":" + str(item[2])
            }
            result.append(content)
            self.store_law_in_memory(content)
        return result

    def get_relevant_laws(self):
        url = "http://172.19.82.198:5014/get_law_name_and_items"
        body = {
            "fact": str(self.fact) + " " + self.problem + " " + "".join(self.claim_list),
            "problem": self.problem,
            "claim_list": self.claim_list
        }
        resp_json = requests.post(url, json=body).json()
        return self._reformat(resp_json)
