# -*- coding: utf-8 -*-
from cmath import log
import logging
import numpy as np
import traceback
import multiprocessing
from config_loader import *
from data_util import cut_words, date_filter, question_filter


def written_correct(inputs):
    return inputs.replace('～', '，').replace('~', '，').replace('并末', '并未').replace('无儿无女', '无儿无女，')


def inputs_reverse(inputs):
    sentence_list = [s for s in re.split('[。？！!?]', inputs) if len(s) > 0]
    sentence_list.reverse()
    return '。'.join(sentence_list)


def single_case_match(inputs, problem, suqiu):
    """
    匹配单独的一个输入(多个句子)，得到特征
    :param inputs: 用户输入的语句
    :return: 返回匹配到的特征和对应的短句
    """
    if suqiu is not None:
        factor_bkw = suqiu_bkw_dict[problem + '_' + suqiu]
        factor_pkw = suqiu_pkw_dict[problem + '_' + suqiu]
        factor_nkw = suqiu_nkw_dict[problem + '_' + suqiu]
    else:
        factor_bkw = problem_bkw_dict[problem]
        factor_pkw = problem_pkw_dict[problem]
        factor_nkw = problem_nkw_dict[problem]
    inputs_factor = {}
    if inputs is None or len(inputs.strip()) == 0:
        return inputs_factor

    inputs = written_correct(inputs)
    inputs = date_filter(inputs)    # 去掉中文年月日
    inputs = question_filter(inputs)    # 去掉疑问词：怎么|如何|吗|么
    inputs = inputs_reverse(inputs)  # 切分句子，逆序后再用。拼接

    factors = []    # inputs匹配上bkw对应的任一个正则中的全部关键词，才把factor加进去
    for factor, keywords_list in factor_bkw.items():
        for keywords in keywords_list:
            if_add = True
            for keyword in keywords:
                if len(re.findall(keyword, inputs)) == 0:
                    if_add = False
                    break
            if if_add:
                factors.append(factor)
                break

    for factor in factors:
        for keyword in factor_nkw[factor]:
            # print(factor, keyword)
            result, flag = match_sentence_by_keyword(inputs, keyword)
            # print(flag)
            if flag == 1:
                inputs_factor[factor] = [result, 1] if factor in no_negative_factor else [result, -1]
                break

        for keyword in factor_pkw[factor]:
            # print(keyword)
            result, flag = match_sentence_by_keyword(inputs, keyword)
            # print(flag)
            if flag == 1:
                if factor not in inputs_factor:
                    inputs_factor[factor] = [result, 1]
                elif sentence_keyword_dict['没有'] in keyword:    # 若本身是否定句，且否定形式（肯定句）已匹配（-1），则重新修改未1
                    inputs_factor[factor] = [result, 1]
                # break # G不break，遍历所有本身是否定句的正则

    return inputs_factor


def match_sentence_by_keyword(sentence, key_word):
    """
    匹配关键词(key_word)。返回正向匹配（1）、负向匹配（-1）或没有匹配（0）
    :param sentence: 一整句话。如"xxx1，xxx2，xxx3，xxxx4。"
    :param key_word: 子案由因子下面的一个key_word
    :return:
    """
    # if sentence.startswith('如') or sentence.startswith('若') or len(re.findall('[如若]' + key_word, sentence)) > 0:
    #     return 0
    # if len(re.findall('(约定|要求|认为|应当|应该|如果).*' + key_word, sentence)) > 0:
    #     return 0
    flag_positive = len(re.findall(key_word, sentence)) > 0
    if not flag_positive or key_word == "()":
        return None, 0

    # 2.匹配负向
    result = re.findall(key_word, sentence)[0][0]
    start_index = sentence.index(result)
    end_index = start_index + len(result)
    while start_index > 0 and sentence[start_index - 1] not in ['，', '。', '；', '：', '？', '！', ',', ';', ':', '?', '!']:
        start_index = start_index - 1
    while end_index < len(sentence) and sentence[end_index] not in ['，', '。', '；', '：', '？', '！', ',', ';', ':', '?', '!']:
        end_index = end_index + 1
    result = sentence[start_index: end_index]
    if sentence_keyword_dict['没有'] not in key_word:
        return result, 1
    else:
        # index = key_word.index('(没|未|不|非|无)[^。；，：,;:？！!?\s]*') + len('(没|未|不|非|无)[^。；，：,;:？！!?\s]*')
        try:
            index = key_word.index('(没有|没|未|不|非|无|未经|怠于)[^。；，：,;:？！!?\\s]*') + len('(没有|没|未|不|非|无|未经|怠于)[^。；，：,;:？！!?\\s]*')
        except:
            logger.error(traceback.format_exc())
            logger.warning(key_word)
            index = key_word.index('(没有|没|未|不|非|无|未经|怠于)')

        tag = 1
        pattern = key_word[index]
        while tag>0:
            index += 1
            pattern += key_word[index]
            if key_word[index]=='(':
                tag += 1
            if key_word[index]==')':
                tag -= 1
        pattern = re.findall('(' + pattern + ')', result)[0][0]
        # pattern = re.findall('(' + key_word.replace(r'(没|未|不|非|无)[^。；，：,;:？！!?\s]*', '') + ')', result)[0][0]
        if pattern == '成年' and '未成年' in result:
            return result, 1

        # 对于存在否定词的情形，调用ltp句法分析包进行句法分析，如果否定词修饰的词在正则里才表示否定意义
        words = cut_words(result)  # 分词 元芳你怎么看
        postags = list(postagger.postag(words))  # 词性标注
        arcs = parser.parse(words, postags)  # 句法分析
        # print(key_word)
        # print(pattern)
        # print(words)
        # print(postags)
        # print([arc.head for arc in arcs])
        # print([arc.relation for arc in arcs])
        for i in range(len(arcs)):
            if words[i] in negative_word_list and arcs[i].relation != 'HED':
                if words[arcs[i].head - 1] in pattern:
                    return result, 1
                if arcs[arcs[i].head - 1].relation == 'VOB' and words[arcs[arcs[i].head - 1].head - 1] in pattern:
                    return result, 1
                if len(pattern) == 1 and (pattern in words[arcs[i].head - 1] or pattern in words[arcs[arcs[i].head - 1].head - 1]):
                    return result, 1
                if words[arcs[i].head - 1] in ['按照']:
                    return result, 1
            if words[i] in negative_word_list and arcs[i].relation == 'HED':
                for k, arc in enumerate(arcs):
                    if arc.relation in ['SBV', 'VOB'] and arc.head == i + 1:
                        if words[k] in pattern or words[k] in ['证据', '证明']:
                            return result, 1
                        if len(pattern) == 1 and pattern in words[k]:
                            return result, 1
            if words[i] in negative_word_list and arcs[arcs[i].head - 1].relation == 'HED' and len(words) > 3:
                for k, arc in enumerate(arcs):
                    if arc.relation in ['SBV', 'VOB'] and arc.head == arcs[i].head:
                        if words[k] in pattern or words[k] in ['证据', '证明']:
                            return result, 1
                        if len(pattern) == 1 and pattern in words[k]:
                            return result, 1
        return result, 0


def multi_processing_data(lines, process_num, problem, suqiu):
    """

    :param process_num:
    :param problem:
    :param num_examples:
    :param num_training:
    :param num_validation:
    :return:
    """
    print("multiprocessing of extract feature.started;process_num:", process_num)
    chunks = build_chunk(lines, chunk_num=process_num - 1)  # 4.1 split data as chunks
    pool = multiprocessing.Pool(processes=process_num)
    for chunk_id, each_chunk in enumerate(chunks):  # 4.2 process each chunk,and save result to file system
        pool.apply_async(get_X, args=(
            each_chunk, problem, suqiu, "tmp_" + str(chunk_id)))  # apply_async
    pool.close()
    pool.join()

    print("allocate work load finished. start map stage.")
    X = []
    for chunk_id in range(process_num):  # 4.3 merge sub file to final file.
        temp_file_name = "tmp_" + str(chunk_id) + ".npy"  # get file name
        X.append(np.load(temp_file_name))
        rm_command = 'rm ' + temp_file_name
        os.system(rm_command)
    X = np.vstack(X)
    print("multiprocessing of extract feature.ended.")
    print("X shape:", X.shape)
    return np.vstack(X)


def build_chunk(lines, chunk_num=4):
    """
    :param lines: total thing
    :param chunk_num: num of chunks
    :return: return chunks but the last chunk may not be equal to chunk_size
    """
    total = len(lines)
    chunk_size = float(total) / float(chunk_num + 1)
    chunks = []
    for i in range(chunk_num + 1):
        if i == chunk_num:
            chunks.append(lines[int(i * chunk_size):])
        else:
            chunks.append(lines[int(i * chunk_size):int((i + 1) * chunk_size)])
    return chunks


def get_X(lines_training, problem, suqiu, target_file):
    """
    匹配输入，得到特征
    :param lines_training:
    :param pattern_list:
    :return:
    """
    try:
        X = []
        if suqiu is not None:
            factors = suqiu_pkw_dict[problem + '_' + suqiu].index
        else:
            factors = problem_pkw_dict[problem].index
        for i, line in enumerate(lines_training):
            match_result = single_case_match(line, problem, suqiu)
            X.append([match_result[f][1] if f in match_result else 0 for f in factors])
        np.save(target_file, X)
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    sentence = '没有打借条'
    print(single_case_match(sentence, '借贷纠纷', None))

    # print(re.findall('(((外出))[^。？！?!]*?((伤|摔|断|跌倒|骨折|死亡|事故|出事|受损|出血|昏迷)))', sentence))

    # words = list(segmentor.segment('驾驶营运客车或校车，载人超过核定人数20％以上的'))  # 分词 元芳你怎么看
    # postags = list(postagger.postag(words))  # 词性标注
    # arcs = parser.parse(words, postags)  # 句法分析
    # print('\t'.join(words))
    # print('\t'.join(postags))
    # print('\t'.join([str(arc.head) for arc in arcs]))
    # print('\t'.join([str(arc.relation) for arc in arcs]))
