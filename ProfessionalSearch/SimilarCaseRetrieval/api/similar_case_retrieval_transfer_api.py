import json

from flask import Flask, request

from LawsuitPrejudgment.lawsuit_prejudgment.core import civil_similar_case
from Utils.http_response import response_successful_result
from ProfessionalSearch.SimilarCaseRetrieval.core import similar_case_retrieval_service as service
from ProfessionalSearch.SimilarCaseRetrieval.core.relevant_cases_search import get_case_search_result
from ProfessionalSearch.SimilarCaseRetrieval.core.narrative_similarity_predict import predict_fn as predict_fn_similar_cases
from typing import List
import logging


from Utils.io import read_json_attribute_value

app = Flask(__name__)

@app.route('/top_k_similar_narrative', methods=["post"])  # "service_type":'ft'
def hello_world():
    # try:
    if 1 == 1:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            fact = in_dict['fact']
            problem = in_dict.get('problem', '')
            claim_list = in_dict.get('claim_list', [])
            logging.info(
                "top_k_similar_narrative.fact:" + fact + ";problem:" + problem + ";claim_list:" + str(claim_list))
            doc_id_list, sim_list, win_los_list, reason_name_list, appeal_name_list, tags_list, keywords = predict_fn_similar_cases(
                fact, problem, claim_list)

            return json.dumps(
                {'dids': doc_id_list, 'sims': sim_list, "winLos": win_los_list, "reasonNames": reason_name_list,
                 "appealNames": appeal_name_list, "tags": tags_list, 'keywords': keywords, 'error_msg': '',
                 'status': 0}, ensure_ascii=False)

        else:
            return json.dumps({"error_msg": "data is None", "status": 1}, ensure_ascii=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8163, debug=True)