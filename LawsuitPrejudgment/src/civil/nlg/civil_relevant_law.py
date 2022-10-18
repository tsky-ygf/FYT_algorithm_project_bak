#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 3/9/2022 12:13 
@Desc    : None
"""
import pandas as pd
import requests
from pypinyin import lazy_pinyin

from ProfessionalSearch.RelevantLaws.api.constants import SEPERATOR_BETWEEN_LAW_TABLE_AND_ID

abandoned_laws = pd.read_csv("data/LawsuitPrejudgment/relevant_laws/已废弃的法律.csv", usecols=["law_name"], encoding="utf-8")["law_name"].tolist()
df_mapping = pd.read_csv("data/LawsuitPrejudgment/relevant_laws/模型输出法条id.csv", encoding="utf-8")


class CivilRelevantLaw:
    def __init__(self, fact, problem, claim_list):
        self.fact = fact
        self.problem = problem
        self.claim_list = claim_list
    #
    # @staticmethod
    # def get_law_id(law_name, law_item):
    #     return "MEMORY_" + "".join(
    #         lazy_pinyin(str(law_name).replace("》", "").replace("《", "").strip())) + "-" + "".join(
    #         lazy_pinyin(str(law_item).strip()))

    @staticmethod
    def _remove_abandoned_laws(law_name_and_items):
        return [law for law in law_name_and_items if law[0] not in abandoned_laws]

    @staticmethod
    def _get_law_id(law_name, law_item):
        return next((row["table_name"] + SEPERATOR_BETWEEN_LAW_TABLE_AND_ID + row["law_id"] for idx, row in df_mapping.iterrows() if str(row["law_name"]) == str(law_name) and str(row["law_item"]) == str(law_item)), None)

    def _reformat(self, law_name_and_items):
        if not law_name_and_items:
            return []

        law_name_and_items.sort(key=lambda x: x[3], reverse=True)
        law_name_and_items = law_name_and_items[:10]
        result = []
        for item in law_name_and_items:
            law_id = self._get_law_id(item[0], item[1])
            if not law_id:
                continue
            content = {
                "law_id": law_id,
                "law_name": item[0],
                "law_item": item[1],
                "law_content": str(item[1]) + ":" + str(item[2])
            }
            result.append(content)
        return result

    def get_relevant_laws(self):
        url = "http://172.19.82.198:5014/get_law_name_and_items"
        body = {
            "fact": str(self.fact) + " " + self.problem + " " + "".join(self.claim_list),
            "problem": self.problem,
            "claim_list": self.claim_list
        }
        resp_json = requests.post(url, json=body).json()
        law_name_and_items = resp_json.get("law_name_and_items", [])
        return self._reformat(self._remove_abandoned_laws(law_name_and_items))
