#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/2 15:42 
@Desc    : None
"""
import json


def response_successful_result(result):
    return json.dumps({"success": True, "error_msg": "", "result": result}, ensure_ascii=False)
