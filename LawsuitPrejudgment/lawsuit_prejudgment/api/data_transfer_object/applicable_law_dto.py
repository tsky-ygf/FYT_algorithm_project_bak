#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 22/8/2022 15:39 
@Desc    : None
"""
from pypinyin import lazy_pinyin


class ApplicableLawDTO:
    def __init__(self, applicable_law):
        self.law_name = applicable_law.get("law_name")
        self.law_item = applicable_law.get("law_item")
        self.law_content = applicable_law.get("law_content")

    @property
    def law_id(self):
        return "-".join(lazy_pinyin(str(self.law_name).replace("》", "").replace("《", "").strip() + str(self.law_item).strip()))

    def to_dict(self):
        return {
            "law_id": self.law_id,
            "law_name": self.law_name,
            "law_item": self.law_item,
            "law_content": self.law_content
        }


class AdministrativeApplicableLawDictCreator:
    @staticmethod
    def create(law):
        law_info = law.get("law_item", "")
        index = str(law_info).find("》") + 1
        applicable_law = {
            "law_name": law_info[:index],
            "law_item": law_info[index:],
            "law_content": law.get("law_content", "")
        }

        return ApplicableLawDTO(applicable_law).to_dict()


class CriminalApplicableLawDictCreator:
    @staticmethod
    def create(law):
        law_name = str(law.get("law_name"))
        applicable_law = {
            "law_name": law_name if law_name.startswith("《") else "《" + law_name + "》",
            "law_item": law.get("law_item"),
            "law_content": law.get("law_content")
        }

        return ApplicableLawDTO(applicable_law).to_dict()
