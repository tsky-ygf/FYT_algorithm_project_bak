#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : server.py
# @Software: PyCharm
import typing

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from LawsuitPrejudgment.service_use import administrative_app_service, civil_app_service, criminal_app_service
from LawsuitPrejudgment.src.common.dialouge_management_parameter import DialogueHistory, DialogueState

app = FastAPI()


class CriminalInput(BaseModel):
    fact: str = Field(example="我偷了同事的3000元")
    question_answers: typing.Dict = Field(example={})

    class Config:
        schema_extra = {
            "example": {
                "fact": "我偷了同事的3000元",
                "question_answers": {}
            }
        }


@app.post("/get_criminal_result")
def _get_criminal_result(param: CriminalInput):
    """
    获取刑事预判的结果。
    本接口即将废弃。
    """
    return criminal_app_service.get_criminal_result(param.fact, param.question_answers)


@app.get("/get_civil_problem_summary")
def _get_civil_problem_summary():
    """
    获取民事预判支持的纠纷类型。

    Returns:
        示例
        {
            "success": true,
            "error_msg": "",
            "value": [{
                "groupName": "婚姻继承",
                "groupList": [{
                    "id": 1533,
                    "problem": "子女抚养"
                }]
            }]
        }
    """
    return civil_app_service.get_civil_problem_summary()


@app.get("/get_template_by_problem_id")
def _get_template_by_problem_id(problem_id: int):
    """
    获取纠纷对应的用户描述模板。

    Returns:
        示例
        {
            "success": true,
            "error_msg": "",
            "value": {
                "template": "男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。"
            }
        }
    """
    return civil_app_service.get_template_by_problem_id(problem_id)


class GetClaimListParam(BaseModel):
    problem_id: int
    fact: str


@app.post("/get_claim_list_by_problem_id")
def _get_claim_list_by_problem_id(param: GetClaimListParam):
    """
    获取纠纷对应的诉求列表。

    Returns:
        示例
        {
            "success": true,
            "error_msg": "",
            "value": [
                {
                    "id": 461,
                    "claim": "请求离婚",
                    "is_recommended": true
                }
            ]
        }
    """
    return civil_app_service.get_claim_list_by_problem_id(param.problem_id, param.fact)


class DialogueInput(BaseModel):
    dialogue_history: DialogueHistory
    dialogue_state: DialogueState


@app.post("/lawsuit_prejudgment")
def _lawsuit_prejudgment(param: DialogueInput):
    """
    诉讼预判的主流程。

    Args:
        dialogue_history: 对话历史。包括用户输入和问答历史。
        dialogue_state: 对话状态。初始化后，后续只需要传递该参数，不用做处理。

    Returns:
        dialogue_history: 对话历史。
        dialogue_state: 对话状态。
        next_action: 下一次行动。包括行动的类型和内容，如提问(aks)、报告(report)等。
    """
    domain = param.dialogue_state.domain
    if domain == "criminal":
        return criminal_app_service.lawsuit_prejudgment(param.dialogue_history, param.dialogue_state)
    if domain == "civil":
        return civil_app_service.lawsuit_prejudgment(param.dialogue_history, param.dialogue_state)
    if domain == "administrative":
        return administrative_app_service.lawsuit_prejudgment(param.dialogue_history, param.dialogue_state)
    raise Exception("传入了不支持的预判类型。")


if __name__ == '__main__':
    uvicorn.run('OnlineServer.LawsuitPrejudgment.server:app', host="0.0.0.0", port=8133, reload=False, workers=1)
