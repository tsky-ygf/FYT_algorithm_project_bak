#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/10 13:14 
@Desc    : None
"""
from flask import Flask, request
import re
from LawsuitPrejudgment.lawsuit_prejudgment.core import civil_similar_case
from Utils.http_response import response_successful_result
from SimilarCaseRetrieval.core import similar_case_retrieval_service as service

app = Flask(__name__)


@app.route('/get_law_document', methods=["get"])
def get_law_document():
    doc_id = request.args.get("doc_id")
    result = service.get_criminal_law_document(doc_id)
    if result:
        return response_successful_result(result)

    law_documents = civil_similar_case.get_civil_law_documents_by_id_list([doc_id])
    if law_documents:
        result = {
            "doc_id": law_documents[0]["doc_id"],
            "html_content": re.sub("href='.+?'", "", law_documents[0]["raw_content"])
        }
    else:
        result = None
    return response_successful_result(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8145, debug=False)