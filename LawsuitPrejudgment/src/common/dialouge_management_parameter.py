#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 18/10/2022 9:20 
@Desc    : None
"""
from pydantic import BaseModel
from typing import Union, List, Dict


class DialogueHistory(BaseModel):
    user_input: Union[str, None]
    question_answers: Union[List, None]


class DialogueState(BaseModel):
    domain: str
    problem: Union[str, None]
    claim_list: Union[List, None]
    other: Union[Dict, None]


class NextAction(BaseModel):
    action_type: str
    content: Dict
