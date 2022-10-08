#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 30/9/2022 17:47 
@Desc    : None
"""
import requests
URL = "http://127.0.0.1:8100"


def get_supported_administrative_types():
    return requests.get(url=URL+"/get_administrative_type").json()


def get_administrative_problem_and_situation_by_type_id(type_id: str):
    return requests.get(url=URL+"/get_administrative_problem_and_situation_by_type_id", params={"type_id": type_id}).json()


def get_administrative_result(type_id, situation):
    return requests.post(url=URL+"/get_administrative_result", json={"type_id": type_id, "situation":situation}).json()
