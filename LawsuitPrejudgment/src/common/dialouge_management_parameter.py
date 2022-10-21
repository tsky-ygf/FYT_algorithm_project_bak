#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 18/10/2022 9:20 
@Desc    : None
"""
from pydantic import BaseModel, Field
from typing import Union, List, Dict


class DialogueHistory(BaseModel):
    user_input: str = Field(default="", description="用户输入的事实描述")
    question_answers: List = Field(default=[], description="用户的问答历史")


class DialogueState(BaseModel):
    domain: str = Field(default="", description="诉讼预判的类型，如'administrative','civil','criminal'")
    problem: str = Field(default="", description="纠纷类型")
    claim_list: List = Field(default=[], description="诉求列表")
    other: Dict = Field(default={}, description="算法的辅助信息，不需要处理")


class NextAction(BaseModel):
    action_type: str = Field(default="", description="下一次行动的类型，如'ask','report'")
    content: Dict = Field(default={}, description="下一次行动的内容")
