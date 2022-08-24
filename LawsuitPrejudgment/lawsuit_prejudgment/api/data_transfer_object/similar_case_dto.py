#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 22/8/2022 15:39 
@Desc    : None
"""
import requests

from LawsuitPrejudgment.lawsuit_prejudgment.constants import CRIMINAL_SIMILIARITY_URL
from SimilarCaseRetrieval.core import similar_case_retrieval_service


class SimilarCaseDTO:
    def __init__(self, similar_case):
        self.doc_id = similar_case.get("doc_id")
        self.similar_rate = similar_case.get("similar_rate")
        self.title = similar_case.get("title")
        self.court = similar_case.get("court")
        self.judge_date = similar_case.get("judge_date")
        self.case_number = similar_case.get("case_number")
        self.tag = similar_case.get("tag")
        self.is_guiding_case = similar_case.get("is_guiding_case")

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "similar_rate": self.similar_rate,
            "title": self.title,
            "court": self.court,
            "judge_date": self.judge_date,
            "case_number": self.case_number,
            "tag": self.tag,
            "is_guiding_case": self.is_guiding_case
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
            "tag": None,
            "is_guiding_case": data.get("is_guiding_case", True)
        }
        return SimilarCaseDTO(similar_case).to_dict()


class CriminalSimilarCaseListCreator:
    @staticmethod
    def create(question):
        resp_json = requests.post(url=CRIMINAL_SIMILIARITY_URL, json={"question": question}).json()
        law_documents = similar_case_retrieval_service.get_criminal_law_document_list(resp_json.get("did_list"))
        if not law_documents:
            return []

        return [
            {
                "doc_id": document["doc_id"],
                "similar_rate": resp_json["sim_list"][idx],
                "title": document["doc_title"],
                "court": document["court"],
                "judge_date": str(document["judge_date"]).replace("发布日期：", ""),
                "case_number": document["case_number"],
                "tag": " ".join(resp_json["tag_list"][idx]),
                "is_guiding_case": True
            } for idx, document in enumerate(law_documents[:10])
        ]
