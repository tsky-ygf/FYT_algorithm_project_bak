#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : inamori1932
# @Site    : 
# @File    : server.py
# @Software: PyCharm
from typing import List, Dict

import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from LawsuitPrejudgment.service_use import administrative_app_service, civil_app_service, criminal_app_service
from LawsuitPrejudgment.src.common.dialouge_management_parameter import DialogueHistory, DialogueState, NextAction

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

    class Config:
        schema_extra = {
            "example": {
                "problem_id": 1531
            }
        }

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

    class Config:
        schema_extra = {
            "example": {
                "problem_id": 1531,
                "fact": "对方经常家暴，我想离婚。"
            }
        }


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
    dialogue_history: DialogueHistory = Field(description="对话历史。包括用户输入和问答历史。")
    dialogue_state: DialogueState = Field(description="对话状态。初始化后，后续只需要传递该参数，不用做处理。")


dialogue_input_examples = {
    "civil": {
        "summary": "民事预判参数示例",
        "value": {
            "dialogue_history": {
                "user_input": "对方经常家暴，我想要离婚。",
                "question_answers": []
            },
            "dialogue_state": {
                "domain": "civil",
                "problem": "婚姻家庭",
                "claim_list": ["请求离婚", "返还彩礼"],
                "other": {}
            }
        }
    },
    "criminal": {
        "summary": "刑事预判参数示例",
        "value": {
            "dialogue_history": {
                "user_input": "我偷了舍友500块。",
                "question_answers": []
            },
            "dialogue_state": {
                "domain": "criminal",
                "problem": "",
                "claim_list": [],
                "other": {}
            }
        }
    },
    "administrative": {
        "summary": "行政预判参数示例",
        "value": {
            "dialogue_history": {
                "user_input": "",
                "question_answers": []
            },
            "dialogue_state": {
                "domain": "administrative",
                "problem": "",
                "claim_list": [],
                "other": {}
            }
        }
    }
}


class DialogueOutput(BaseModel):
    success: bool = Field(description="是否成功调用")
    dialogue_history: DialogueHistory = Field(description="对话历史。与入参的值相同。")
    dialogue_state: DialogueState = Field(description="对话状态。经过算法更新后的对话状态，与入参的值不同。")
    next_action: NextAction = Field(description="下一次行动的信息。")


@app.post("/lawsuit_prejudgment", summary="诉讼预判的主流程", response_model=DialogueOutput)
def _lawsuit_prejudgment(param: DialogueInput = Body(examples=dialogue_input_examples)):
    """
    诉讼预判的主流程。

    请求参数:


    | Param            | Type | Description |
    |------------------|------|-------------|
    | dialogue_history | Dict | 对话历史        |
    | dialogue_state   | Dict | 对话状态        |

    dialogue_history的内容如下:
    * user_input: string, 事实描述
    * question_answers: List, 问答记录
      * question: string, 问题
      * candidate_answers: List, 候选项
      * question_type: string, 问题类型，如“single”代表单选，“multiple”代表多选
      * user_answer: List, 用户答案
      * other: Dict, 其他辅助信息，不用做处理。

    dialogue_state的内容如下:
    * domain: string, 诉讼预判的类型，如'civil'代表民事, 'criminal'代表刑事, 'administrative'代表行政
    * problem: string, 纠纷类型
    * claim_list: List, 诉求列表
    * other: Dict, 其他辅助信息，不用做处理。

    响应参数:


    | Param            | Type    | Description |
    |------------------|---------|-------------|
    | success          | boolean | 是否成功调用      |
    | dialogue_history | Dict    | 对话历史        |
    | dialogue_state   | Dict    | 对话状态        |
    | next_action      | Dict    | 下一次行动的信息    |

    next_action的内容如下:

      * action_type: string, 下一次行动的类型, 如'ask'代表提问, 'report'代表产生报告

      * content: Dict, 下一次行动的内容

        如果是ask,内容为
          * question: string, 问题
          * candidate_answers: List, 候选项
          * question_type: string, 问题类型，如“single”代表单选，“multiple”代表多选
          * other: Dict, 其他辅助信息，不用做处理。

        如果是report,内容为
          * report: List, 预判报告内容
          * similar_case: List, 相似案例
          * applicable_law: List, 相关法条

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
