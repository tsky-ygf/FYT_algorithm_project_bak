#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 17:45 
@Desc    : None
"""
import requests

from LawsuitPrejudgment.src.civil.lawsuit_prejudgment.api.data_transfer_object.prejudgment_report_dto import \
    CriminalReportDTO
from LawsuitPrejudgment.src.criminal.criminal_prejudgment import CriminalPrejudgment

criminal_config = {
    "log_level": "info",
    "prejudgment_type": "criminal",
    "anyou_identify_model_path": "model/gluon_model/accusation",
}
criminal_pre_judgment = CriminalPrejudgment(**criminal_config)


def _get_criminal_result(fact, question_answers):
    if question_answers == {}:
        criminal_pre_judgment.init_content()
    # 改变question_answers的格式：将情形作为键值。这样CriminalPrejudgment才能处理。
    use_qa = {}
    for question, answer in question_answers.items():
        circumstance, info = criminal_pre_judgment.get_circumstance_of_question(question)
        use_qa[circumstance] = info
        use_qa[circumstance]["usr_answer"] = answer

    criminal_pre_judgment(fact=fact, question_answers=use_qa)

    while "report_result" not in criminal_pre_judgment.content:
        next_question = criminal_pre_judgment.get_next_question()
        return {"next_question": next_question}
    return {"result": criminal_pre_judgment.content["report_result"]}


def get_criminal_result(fact, question_answers):
    result = _get_criminal_result(fact, question_answers)
    middle_layer = CriminalResultMiddleLayer(fact, question_answers)
    middle_layer.response_from_criminal_server = result
    result = middle_layer.get_criminal_result()
    return CriminalReportDTO(result).to_dict()


class CriminalResultMiddleLayer:
    """
        中间层:
        用来处理外部网络接口(OnlineServer/LawsuitPrejudgment/server.py)和内部刑事预判(criminal_server)的参数差异。
    """

    def __init__(self, fact, question_answers):
        self.body = {"fact": fact, "question_answers": question_answers}
        self.response_from_criminal_server = None

    # def _get_response_from_criminal_server(self):
    #     self.response_from_criminal_server = requests.post(url=self.criminal_server_url, json=self.body).json()

    def _have_next_question(self):
        return "result" not in self.response_from_criminal_server

    def _update_next_question(self, criminal_result):
        info = next(question_info for circumstance, question_info in
                    self.response_from_criminal_server["next_question"].items())
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
        # self._get_response_from_criminal_server()
        print(self.response_from_criminal_server)
        return self._generate_criminal_result()


if __name__ == "__main__":
    res = get_criminal_result(fact="我偷了邻居3000元怎么办？",
                              question_answers={}, )
    # question_answers={"盗窃人年龄未满十六周岁或者精神状态不正常？:是;否": "否"})
    print(res)
