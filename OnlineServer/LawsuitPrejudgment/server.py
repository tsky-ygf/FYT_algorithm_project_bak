#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : server.py
# @Software: PyCharm
from typing import List, Dict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from LawsuitPrejudgment.service_use import administrative_app_service, civil_app_service, criminal_app_service
from LawsuitPrejudgment.src.common.dialouge_management_parameter import DialogueHistory, DialogueState

app = FastAPI()


class ProblemItem(BaseModel):
    id: int = Field(description="纠纷id")
    problem: str = Field(description="纠纷名称")


class ProblemGroup(BaseModel):
    groupName: str = Field(description="组名")
    groupList: List[ProblemItem] = Field(description="数据列表")


class GetCivilProblemSummaryResult(BaseModel):
    success: bool = Field(description="是否成功调用")
    error_msg: str = Field(description="失败时的错误信息")
    value: List[ProblemGroup] = Field(description="数据列表")


@app.get("/get_civil_problem_summary", summary="获取民事预判支持的纠纷类型", response_model=GetCivilProblemSummaryResult)
def _get_civil_problem_summary():
    """
    获取民事预判支持的纠纷类型。

    响应参数:

    | Param     | Type    | Description  |
    |-----------|---------|--------------|
    | success   | boolean | 是否成功调用       |
    | error_msg | string  | 失败时的错误信息     |
    | value     | List    | 数据列表         |

    value的内容如下:

    * groupName: string, 组名

    * groupList: List, 数据列表

      * id: int, 纠纷id

      * problem: string, 纠纷名称
    """
    return civil_app_service.get_civil_problem_summary()


class GetTemplateByProblemIdInput(BaseModel):
    problem_id: int = Field(description="纠纷id", example=1531)


class TemplateValue(BaseModel):
    template: str = Field(description="用户描述模板")


class GetTemplateByProblemIdResult(BaseModel):
    success: bool = Field(description="是否成功调用")
    error_msg: str = Field(description="失败时的错误信息")
    value: TemplateValue = Field(description="数据字典")


@app.post("/get_template_by_problem_id", summary="获取纠纷对应的用户描述模板", response_model=GetTemplateByProblemIdResult)
def _get_template_by_problem_id(param: GetTemplateByProblemIdInput):
    """
    获取纠纷对应的用户描述模板。

    请求参数:

    | Param      | Type | Description |
    |------------|------|-------------|
    | problem_id | int  | 纠纷id        |

    响应参数:

    | Param     | Type    | Description |
    |-----------|---------|-------------|
    | success   | boolean | 是否成功调用      |
    | error_msg | string  | 失败时的错误信息    |
    | value     | Dict    | 数据字典        |

    value的内容如下:

    * template: string, 用户描述模板

    """
    return civil_app_service.get_template_by_problem_id(param.problem_id)


class GetClaimListByProblemIdInput(BaseModel):
    problem_id: int = Field(description="纠纷id", example=1531)
    fact: str = Field(description="用户输入的事实描述", example="对方经常家暴，我想离婚。")


class ClaimItem(BaseModel):
    id: int = Field(description="诉求id")
    claim: str = Field(description="诉求名称")
    is_recommended: bool = Field(description="是否推荐该诉求。如果为true，则前端把该诉求设置为选中。")


class GetClaimListByProblemIdResult(BaseModel):
    success: bool = Field(description="是否成功调用")
    error_msg: str = Field(description="失败时的错误信息")
    value: List[ClaimItem] = Field(description="数据列表")


@app.post("/get_claim_list_by_problem_id", summary="获取纠纷对应的诉求列表", response_model=GetClaimListByProblemIdResult)
def _get_claim_list_by_problem_id(param: GetClaimListByProblemIdInput):
    """
    获取纠纷对应的诉求列表。

    请求参数:


    | Param      | Type   | Description |
    |------------|--------|-------------|
    | problem_id | int    | 纠纷id        |
    | fact       | string | 事实描述        |


    响应参数:


    | Param     | Type    | Description |
    |-----------|---------|-------------|
    | success   | boolean | 是否成功调用      |
    | error_msg | string  | 失败时的错误信息    |
    | value     | List    | 数据列表        |

    value的内容如下:

      * id: int, 诉求id

      * claim: string, 诉求名称

      * is_recommended: boolean, 是否推荐该诉求，示例：true。如果为true，则前端把该诉求设置为选中。
      
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
