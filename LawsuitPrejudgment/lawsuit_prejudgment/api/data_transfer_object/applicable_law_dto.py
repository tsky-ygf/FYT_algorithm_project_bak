#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 22/8/2022 15:39 
@Desc    : None
"""


class ApplicableLawDTO:
    def __init__(self, applicable_law):
        self.law_name = applicable_law.get("law_name")
        self.law_item = applicable_law.get("law_item")
        self.law_content = applicable_law.get("law_content")

    def to_dict(self):
        return {
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
