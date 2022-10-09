#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 17:45 
@Desc    : None
"""
import requests

from LawsuitPrejudgment.lawsuit_prejudgment.api.data_transfer_object.prejudgment_report_dto import CriminalReportDTO


def get_criminal_result(fact, question_answers, factor_sentence_list):
    # 用时大约3秒
    # body = {
    #     "fact": fact,
    #     "question_answers": question_answers,
    #     "factor_sentence_list": factor_sentence_list
    # }
    # return requests.post(url="http://127.0.0.1:8100/get_criminal_result", json=body).json()
    # 用时大约10秒 TODO:分析和改进耗时长的原因。
    criminal_result = CriminalResultMiddleLayer(fact, question_answers).get_criminal_result()
    return CriminalReportDTO(criminal_result).to_dict()


class CriminalResultMiddleLayer:
    """
        中间层:
        用来处理外部网络接口(OnlineServer/LawsuitPrejudgment/server.py)和内部刑事预判(criminal_server)的参数差异。
    """
    def __init__(self, fact, question_answers):
        self.body = {"fact": fact, "question_answers": question_answers}
        self.criminal_server_url = "http://127.0.0.1:5081/criminal_prejudgment"
        self.response_from_criminal_server = None

    def _get_response_from_criminal_server(self):
        self.response_from_criminal_server = requests.post(url=self.criminal_server_url, json=self.body).json()

    def _have_next_question(self):
        return "result" not in self.response_from_criminal_server

    def _update_next_question(self, criminal_result):
        info = next(question_info for circumstance, question_info in self.response_from_criminal_server["next_question"].items())
        criminal_result["question_next"] = str(info.get("question")) + ":" + str(info.get("answer")).replace("|", ";")
        criminal_result["question_type"] = ("1" if info.get("multiplechoice", 0) == 0 else "2")

    def _update_result(self, criminal_result):
        report_result = self.response_from_criminal_server["result"]
        criminal_result["support"] = ("敬请期待" not in report_result)
        if not criminal_result["support"]:
            criminal_result["result"] = {
                "unsupport_reason": report_result
            }
            return

        criminal_result["result"] = {
                "crime": report_result.get("涉嫌罪名", ""),
                "case_fact": report_result.get("案件事实", ""),
                "reason_of_evaluation": report_result.get("评估理由", ""),
                "legal_advice": report_result.get("法律建议", ""),
                "similar_case": report_result.get("相关类案", []),
                "applicable_law": report_result.get("法律依据", "")
            }
        pass

    def _generate_criminal_result(self):
        criminal_result = {
            "success": True,
            "error_msg": "",
            "question_asked": self.body["question_answers"],
            "question_next": None,
            "question_type": "1",
            "factor_sentence_list": [],
            "support": True,
            "result": None
        }

        if self._have_next_question():
            self._update_next_question(criminal_result)
        else:
            self._update_result(criminal_result)

        return criminal_result

    def get_criminal_result(self):
        self._get_response_from_criminal_server()
        return self._generate_criminal_result()
