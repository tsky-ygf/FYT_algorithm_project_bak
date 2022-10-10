#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 15:40 
@Desc    : None
"""
from LawsuitPrejudgment.src.civil.common import single_case_match


def test_single_case_match():
    """
        发现:
            考虑"借款人死亡"这个特征。对比【[借贷][死]】和【[借贷]】【[死]】
            1)"借"和"去世"直接没有标点符号，两种模式都能匹配出来。
            fact = "我朋友向我借了二十万块钱后来他去世了，我像大孩子要钱"
            2)用句号分割。
            【[A]】【[B]】未考虑多句(。等分隔)的匹配。
            对于A。B。两种模式都匹配不出来。
            fact = "我朋友向我借了二十万块钱。后来他去世了，我像大孩子要钱"
            3)用逗号分割。
            【[A][B]】未考虑句内(，等分割)的匹配。
            对于A，B。【[A][B]】匹配不出来。【[A]】【[B]】能匹配出来。
            # fact = "我朋友向我借了二十万块钱，后来他去世了。我像大孩子要钱"

            相应的处理逻辑，在config_loader.get_factor_positive_keyword()
    """
    problem = "借贷纠纷"
    fact = "我朋友向我借了二十万块钱，后来他去世了。我像大孩子要钱"

    result = single_case_match(fact, problem, None)

    print("#"*50)
    print(result)
    assert result
    assert "借款人借款后死亡" in result or "借款人死亡" in result
    assert result["借款人借款后死亡"][1] == 1 or result["借款人死亡"][1] == 1
