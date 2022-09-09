#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 23/8/2022 15:06 
@Desc    : None
"""
from ProfessionalSearch.SimilarCaseRetrieval.repository.similar_case_retrieval_repository import get_criminal_law_documents_by_id_list


def test_get_criminal_law_documents_by_id_list():
    id_list = ["24dbed45-904d-4992-aea7-a82000320181"]
    criminal_law_documents = get_criminal_law_documents_by_id_list(id_list)

    print(criminal_law_documents)
    assert criminal_law_documents
    assert len(criminal_law_documents) == len(id_list)
    assert criminal_law_documents[0]["doc_id"] == id_list[0]
    assert criminal_law_documents[0]["doc_title"] == "李某汉、黄某娟走私、贩卖、运输、制造毒品一审刑事判决书"
