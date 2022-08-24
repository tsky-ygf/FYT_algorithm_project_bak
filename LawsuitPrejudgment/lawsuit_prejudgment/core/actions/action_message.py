#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/7/29 16:02 
@Desc    : None
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionMessage:
    """ 执行动作需要的数据。 """
    pass
