#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/10 13:14 
@Desc    : None
"""
from flask import Flask, request
import re

from LawsuitPrejudgment.lawsuit_prejudgment.constants import CIVIL_SIMILAR_CASE_ID_PREFIX
from LawsuitPrejudgment.lawsuit_prejudgment.core import civil_similar_case
from Utils.http_response import response_successful_result
from ProfessionalSearch.SimilarCaseRetrieval.core import similar_case_retrieval_service as service

app = Flask(__name__)


@app.route('/get_law_document', methods=["get"])
def get_law_document():
    doc_id = str(request.args.get("doc_id"))
    original_doc_id = doc_id
    if doc_id.startswith(CIVIL_SIMILAR_CASE_ID_PREFIX):
        doc_id = doc_id[len(CIVIL_SIMILAR_CASE_ID_PREFIX):]
        law_documents = civil_similar_case.get_civil_law_documents_by_id_list([doc_id])
        if law_documents:
            result = {
                "doc_id": law_documents[0]["doc_id"],
                "html_content": re.sub("href='.+?'", "", law_documents[0]["raw_content"])
            }
        else:
            result = None
    else:
        table_name, doc_id = doc_id.split("_SEP_")
        if table_name == "judgment_xingshi_data":
            result = service.get_criminal_law_document(doc_id)
        elif table_name == "judgment_minshi_data":
            result = service.get_civil_law_document(doc_id)
        else:
            result = service.get_law_document_in_db(doc_id, table_name)

    # 为配合APP的收藏功能，将result的doc_id置为分隔符+数据库中相应的id。也就是入参的doc_id。
    if result:
        result["doc_id"] = original_doc_id

    return response_successful_result(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8145, debug=False)