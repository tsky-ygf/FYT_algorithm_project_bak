#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 3/9/2022 12:12 
@Desc    : None
"""
import requests


class CivilSimilarCase:
    def __init__(self, fact, problem, claim_list):
        self.fact = fact
        self.problem = problem
        self.claim_list = claim_list

    def get_similar_cases(self):
        url = "http://172.19.82.198:5011/top_k_similar_narrative"
        body = {
            "fact": self.fact,
            "problem": self.problem,
            "claim_list": self.claim_list
        }
        resp_json = requests.post(url, json=body).json()
        return resp_json