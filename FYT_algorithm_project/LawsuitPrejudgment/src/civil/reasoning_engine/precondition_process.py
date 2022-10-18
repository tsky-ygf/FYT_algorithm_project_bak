# -*- coding: utf-8 -*-
from LawsuitPrejudgment.src.civil.utils.config_loader import *


def prediction_suqiu_filter(predictions, suqiu_result):
    result = []
    for prediction in predictions:
        add = True
        for key, value in prediction.items():
            if key.startswith('诉求'):
                suqiu = key.split('_')[1]
                if suqiu in suqiu_result and suqiu_result[suqiu] != value:
                    add = False
                    break
        if add:
            result.append(prediction)
    return result


def prediction_factor_result(ps, factor, suqiu_factor, question_answers, factor_sentence_list):
    result = 0
    if factor in suqiu_factor:
        result = suqiu_factor[factor]   # 诉求配置logic_suqiu_factor的默认值
    if factor in factor_sentence_list:
        result = factor_sentence_list[factor][1]    # 用户输入的特征匹配结果
    if factor in factor_question_dict[ps]:  # 如果该factor已经问过，result取用户的回答结果
        question = factor_question_dict[ps][factor]
        if question in question_answers:
            answers = question_answers[question].split(';')
            for answer in factor_answer_dict[ps][factor].split('|'):
                if answer in answers:
                    result = 1
            if result == 0:
                result = -1
    return result


def get_next_suqiu_or_factor(ps, suqiu_result, suqiu_factor, question_answers, factor_sentence_list):
    # 无前置的诉求特征的，返回suqiu
    # 有的：过滤诉求后空的返回None，
    #       不空的： 某个前置条件不包含特征，返回suqiu
    #               某个特征prediction_factor_result为0，返回该factor
    # 否则返回None
    problem, suqiu = ps.split('_')
    if ps not in logic_ps_prediction:   # 如果problem没有前置诉求或前置特征，返回诉求
        return ('suqiu', suqiu)

    predictions = logic_ps_prediction[ps]
    # 过滤predictions中，suqiu_result中有，并且suqiu_result[suqiu]!=prediction值的suqiu
    predictions = prediction_suqiu_filter(predictions, suqiu_result)
    if len(predictions) == 0:   # 如果前置条件都不满足返回None
        return None

    next = None
    for prediction in predictions:
        satisfied = True
        for key, value in prediction.items():
            if key.startswith('特征'):
                factor = key.split('_')[1]
                result = prediction_factor_result(ps, factor, suqiu_factor, question_answers, factor_sentence_list)
                if result == 0:  # 既没有默认值，也没有匹配用户输入，也没有问过问题
                    next = factor
                    satisfied = False
                elif result != value:   # 特征结果不等于前置条件的值
                    satisfied = False
        if satisfied:   # 若某个prediction中不包含特征直接返回诉求
            return ('suqiu', suqiu)
    if next is not None:    # 某个prediction的factor，既没有默认值，也没有匹配用户输入，也没有问过问题
        return ('factor', next)
    else:   # 所有factor，特征结果不等于前置条件的值
        return None
