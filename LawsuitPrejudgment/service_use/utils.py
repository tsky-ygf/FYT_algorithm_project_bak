#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 8/10/2022 10:25 
@Desc    : None
"""


def successful_response(result, additional_args_dict=None):
    body = {"success": True, "error_msg": "", "result": result}
    if additional_args_dict:
        body.update(additional_args_dict)
    return body


def failed_response(error_message):
    return {"success": False, "error_msg": error_message, "result": None}
