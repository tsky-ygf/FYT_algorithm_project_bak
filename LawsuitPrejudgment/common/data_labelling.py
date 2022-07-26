# -*- coding: utf-8 -*-
import re
from LawsuitPrejudgment.common.config_loader import anyou_ps_dict, ps_positive_keyword, ps_negative_keyword, ps_panjue_keyword


##############################################################################################################################################
#
# 裁判文书y值打标
#
##############################################################################################################################################


def _sentence_keyword_match(positive_keywords, negative_keywords, sentence_list, func):
    """
    基于正负向关键词进行打标，如果有一个分句同时满足存在关键词并且不满足3个排除关键词，则返回1
    匹配到正向的了，还要看有没有负向的，没有负向的就直接返回1。我这边默认了1为空，23都为空
    :param row:
    :param sentence_list:
    :param func:
    :return:
    """
    for sentence in sentence_list:
        # 匹配正向关键词，匹配到则继续匹配负向关键词，匹配不到则跳到下一句
        if not func(positive_keywords, sentence):
            continue
        # 检查负向关键词是否为空，空则返回1
        if negative_keywords is None or len(negative_keywords.strip()) == 0:
            return 1
        # 匹配负向关键词，未匹配到则返回1
        if not func(negative_keywords, sentence):
            return 1
    return 0


def get_panjue_label_from_config(anyou, panjue_sentences):
    """
    根据诉请和事实理由，解析出诉求类型
    :param panjue_sentences: 判决内容
    :param anyou: 案由
    :param suqing_sentences: 诉求内容
    :return:
    """
    # 1.筛选诉求
    pses = anyou_ps_dict[anyou]
    panjue_sentence_list = re.split('[。；]', panjue_sentences)

    panjue_label = {}
    for ps in pses:
        positive_keywords = ps_positive_keyword[ps]
        if ps in ps_panjue_keyword:
            positive_keywords = '(' + positive_keywords + '|' + ps_panjue_keyword[ps] + ')'
        negative_keywords = None
        if ps in ps_negative_keyword:
            negative_keywords = ps_negative_keyword[ps]
        panjue_label[ps] = _sentence_keyword_match(positive_keywords, negative_keywords, panjue_sentence_list, check_pattern_with_panjue_sentence)

    return panjue_label


def get_suqiu_label_from_config(anyou, suqing_sentences):
    """
    根据诉请和事实理由，解析出诉求类型.
    默认suqiu_anyou_list可以从
    :param panjue_sentences: 判决
    :param anyou: 案由
    :param suqing_sentences: 诉请和事实理由
    :return:
    """
    # 提取诉讼请求
    pses = anyou_ps_dict[anyou]
    suqing_sentence_list = re.split('[。；]', suqing_sentences)

    suqiu_label = {}
    for ps in pses:
        positive_keywords = ps_positive_keyword[ps]
        negative_keywords = None
        if ps in ps_negative_keyword:
            negative_keywords = ps_negative_keyword[ps]
        suqiu_label[ps] = _sentence_keyword_match(positive_keywords, negative_keywords, suqing_sentence_list, check_pattern_with_suqing_sentence)

    return suqiu_label


def check_pattern_with_panjue_sentence(pattern, panjue_sentence, filter_word=None):
    """
    使用句式匹配判决：遍历每一个分句，如果任何一个分句有正向匹配，不存在负向匹配且没有驳回，那么认为匹配到了；否则，没有匹配到
    :param pattern:
    :param panjue_sentence:
    :param filter_word: 过滤词
    :return:
    """
    # print("pattern:",pattern,";panjue_sentence:",panjue_sentence)
    panjuan0 = [x.group() for x in re.finditer(pattern, panjue_sentence)]
    panjuan = []
    if len(panjuan0) > 0:
        for panjue_i in panjuan0:
            temp = panjue_i
            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            panjuan.append(temp)

            temp = panjue_i
            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            panjuan.append(temp)

    flag_pos = len(panjuan) > 0

    flag_neg3 = len(re.findall('驳回.*' + pattern, panjue_sentence)) > 0
    flag_neg4 = len(re.findall(pattern + '.*不予', panjue_sentence)) > 0
    flag_neg5 = len(re.findall('不准.*' + pattern, panjue_sentence)) > 0
    # print(panjuan, flag_pos, flag_neg3, flag_neg4, flag_neg5)
    if flag_pos and not flag_neg3 and not flag_neg4 and not flag_neg5 and not (
            filter_word is not None and filter_word in panjue_sentence):
        return True
    return False


def check_pattern_with_suqing_sentence(pattern, suqing_sentence, filter_word=None):
    """
    使用句式匹配诉求：
    :param pattern:
    :param suqing_sentence: 诉求和事实
    :param filter_word: 过滤词
    :return:
    """
    # print("pattern:",pattern,";suqing_sentence:",suqing_sentence)
    panjuan0 = [x.group() for x in re.finditer(pattern, suqing_sentence)]
    panjuan = []
    if len(panjuan0) > 0:
        for panjue_i in panjuan0:
            temp = panjue_i
            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            panjuan.append(temp)

            temp = panjue_i
            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            panjuan.append(temp)

    # print(panjuan0, panjuan)
    flag_pos = len(panjuan) > 0

    if flag_pos and not (filter_word is not None and filter_word in suqing_sentence):
        # print("suqing:",suqing,";flag_pos:",flag_pos,";pattern:",pattern)
        return True
    return False


def get_label_string(problem, anyou, panjue_sentences, suqing_sentences):
    """
    对DataFrame每行数据进行打标
    :param row: DataFrame行
    :return: 打标结果，字符串
    """
    # 诉求打标
    suqiu_labels = get_suqiu_label_from_config(anyou, suqing_sentences)
    # 判决打标
    panjue_labels = get_panjue_label_from_config(anyou, panjue_sentences)

    result = []
    for ps, value in panjue_labels.items():
        if not ps.startswith(problem+'_'):
            continue
        if ps in suqiu_labels and suqiu_labels[ps] == 1:
            result.append(ps + ':' + str(int(value)))
        else:
            result.append(ps + ':-1')
    return ';'.join(result)


if __name__=='__main__':
    print(get_suqiu_label_from_config('金融借款合同纠纷', '请求法院判令：1、被告连带偿还原告借款本息89282.76元，并向原告支付律师代理费4464元；2、确认原告对被告自有的位于桂平市垌心乡垌心街的一宗地产［国有土地使用证证号：浔国用第2089号，地号：280101257，土地他项权证证号：浔他项第378号］享有优先受偿权；3、本案诉讼费用由被告承担。'))
