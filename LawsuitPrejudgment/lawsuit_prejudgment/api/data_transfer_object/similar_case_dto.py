#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 22/8/2022 15:39 
@Desc    : None
"""


class SimilarCaseDTO:
    def __init__(self, similar_case):
        self.doc_id = similar_case.get("doc_id")
        self.similar_rate = similar_case.get("similar_rate")
        self.title = similar_case.get("title")
        self.court = similar_case.get("court")
        self.judge_date = similar_case.get("judge_date")
        self.case_number = similar_case.get("case_number")
        self.tag = similar_case.get("tag")

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "similar_rate": self.similar_rate,
            "title": self.title,
            "court": self.court,
            "judge_date": self.judge_date,
            "case_number": self.case_number,
            "tag": self.tag
        }


class AdministrativeSimilarCaseDictCreator:
    @staticmethod
    def create(data):
        similar_case = {
            "doc_id": None,
            "similar_rate": 1,
            "title": data.get("content"),
            "court": None,
            "judge_date": None,
            "case_number": None,
            "tag": None
        }
        return SimilarCaseDTO(similar_case).to_dict()
