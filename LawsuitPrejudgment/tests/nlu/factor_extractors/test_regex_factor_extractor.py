#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 15:40 
@Desc    : None
"""
from LawsuitPrejudgment.common import single_case_match


def test_single_case_match():
    problem = "借贷纠纷"
    # fact = "我朋友向我借了二十万块钱，后来他去世了，我像大孩子要钱"
    fact = "我朋友向我借了二十万块钱，后来他去世了。我像大孩子要钱"

    result = single_case_match(fact, problem, None)

    print("#"*50)
    print(result)
    assert result
    assert "借款人借款后死亡" in result
    assert result["借款人借款后死亡"][1] == 1
