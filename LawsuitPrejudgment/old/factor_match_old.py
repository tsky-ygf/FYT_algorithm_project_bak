# -*- coding: utf-8 -*-
import re
import numpy as np
import traceback
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../common'))

from common import config_path, suqiu_keyword_dict, problem_keyword_dict
from pyltp import Parser, Segmentor, Postagger, NamedEntityRecognizer


negative_word_list = ['不', '未', '没', '没有', '无', '非', '并未', '不再', '不能', '无法', '不足以', '不存在', '不能证明', '不认可','尚未', '不行', '没法', '没发', '无发'] # 否定词列表
negative_word = '(' + '|'.join(negative_word_list) + ')' # 否定词正则
no_negative_factor = ['被赡养方', '赡养方', '父方', '母方'] # 没有否定意义的特征


LTP_DATA_DIR = '../model/ltp'  # ltp模型目录的路径
seg_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性模型的路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体模型的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`


# 加载模型
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(seg_model_path, config_path+'userdict.txt')  # 加载模型
postagger = Postagger()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型
recognizer = NamedEntityRecognizer()  # 初始化实例
recognizer.load(ner_model_path)  # 加载模型
parser = Parser()  # 初始化实例
parser.load(par_model_path)


def written_correct(inputs):
    return inputs.replace('并末', '并未').replace('无儿无女', '无儿无女，')


def date_filter(inputs):
    inputs = re.sub('[\d一二三四五六七八九零]{2,4}年[\d一二三四五六七八九零]{1,2}月[\d一二三四五六七八九零]{1,2}日', '', inputs)
    inputs = re.sub('[\d一二三四五六七八九零]{2,4}年[\d一二三四五六七八九零]{1,2}月', '', inputs)
    inputs = re.sub('[\d一二三四五六七八九零]{1,2}月[\d一二三四五六七八九零]{1,2}日', '', inputs)
    inputs = re.sub('[\d一二三四五六七八九零]{4}年', '', inputs)
    return inputs


def single_case_match(inputs, problem, suqiu):
    """
    匹配单独的一个输入(多个句子)，得到特征
    :param inputs: 用户输入的语句
    :return: 返回匹配到的特征和对应的短句
    """
    if suqiu is not None:
        factor_keywords = suqiu_keyword_dict[problem+'_'+suqiu]
    else:
        factor_keywords = problem_keyword_dict[problem]
    sentence_factor = {}
    if inputs is None or len(inputs.strip()) == 0:
        return sentence_factor

    inputs = written_correct(inputs)
    inputs = date_filter(inputs)
    sentence_list = [s for s in re.split('[。；，：,;:？！!?\s]', written_correct(inputs)) if len(s) > 0]
    sentence_list.reverse()

    for factor, keyword_list in factor_keywords.items():
        for keyword in keyword_list:
            # print(keyword)
            if '方' in keyword:
                keyword = '(' + keyword + '|' + keyword.replace('方', '⽅') + ')'
            if '父' in keyword:
                keyword = '(' + keyword + '|' + keyword.replace('父', '⽗') + ')'
            keyword = '(' + keyword + ')'
            flag = 0
            if len(re.findall(keyword.replace('.*', '[^。；，：,;:\s]*'), inputs)) == 0:
                continue
            for index, sentence in enumerate(sentence_list):  # 针对每个句子去匹配
                pattern, flag = match_sentence_by_keyword(sentence, keyword)
                if flag==1 or flag==-1:
                    # print(keyword)
                    sentence_factor[factor] = [pattern, 1, index] if factor in no_negative_factor else [pattern, flag, index]
                    break
            if flag==1 or flag==-1:
                break

    sentence_factor = sorted(sentence_factor.items(), key=lambda x: x[1][2], reverse=True)
    sentence_factor = {sf[0]: [sf[1][0], sf[1][1]] for sf in sentence_factor}
    return sentence_factor


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
    # 2.匹配负向
    if not flag_positive:
        return None, 0
    pattern = re.findall(key_word, sentence)[0][0]
    if len(re.findall(negative_word + '[^。；，：,;:？！!?\s]*' + key_word, sentence)) > 0 \
            or len(re.findall(key_word + '[^。；，：,;:？！!?\s]*' + negative_word, sentence)) > 0:
        if negative_match(key_word, sentence) == -1:
            return pattern, -1
    if len(re.findall(negative_word, key_word)) == 0 and '[^。？！?!]*' in key_word:
        kl = key_word.split('[^。？！?!]*')
        for i in range(len(kl) - 1):
            kw = '[^。？！?!]*'.join([k + '[^。？！?!]*' + negative_word if i == j else k for j, k in enumerate(kl)])
            print(kw)
            if len(re.findall(kw, sentence)) > 0 and negative_match(key_word, sentence) == -1:
                return pattern, -1
    if len(re.findall(negative_word, key_word)) == 0 and '[^。；，：,;:？！!?\s]*' in key_word:
        kl = key_word.split('[^。；，：,;:？！!?\s]*')
        for i in range(len(kl) - 1):
            kw = '[^。；，：,;:？！!?\s]*'.join([k + '[^。；，：,;:？！!?\s]*' + negative_word if i == j else k for j, k in enumerate(kl)])
            if len(re.findall(kw, sentence)) > 0 and negative_match(key_word, sentence) == -1:
                return pattern, -1
    return pattern, 1


# def negative_match(sentence):
#
#     words = jieba.lcut(sentence)
#     for word in words:
#         if word in negative_word_list:
#             return -1
#     return 1


def negative_match(key_word, sentence):
    pattern = re.findall(key_word, sentence)[0][0]
    words = list(segmentor.segment(sentence))  # 分词 元芳你怎么看
    postags = list(postagger.postag(words))  # 词性标注
    arcs = parser.parse(words, postags)  # 句法分析
    # print(key_word)
    # print(pattern)
    # print(words)
    # print(postags)
    # print([arc.head for arc in arcs])
    # print([arc.relation for arc in arcs])
    for i in range(len(arcs)):
        if words[i] in negative_word_list and words[i] not in key_word and arcs[i].relation != 'HED':
            if words[arcs[i].head - 1] in pattern:
                return -1
            if arcs[arcs[i].head - 1].relation == 'VOB' and words[arcs[arcs[i].head - 1].head - 1] in pattern:
                return -1
        if words[i] in negative_word_list and words[i] not in key_word and arcs[i].relation == 'HED':
            for k, arc in enumerate(arcs):
                if arc.relation in ['SBV', 'VOB'] and arc.head == i + 1 and (words[k] in pattern or words[k] in ['证据', '证明']):
                    return -1
        if words[i] in negative_word_list and words[i] not in key_word and arcs[arcs[i].head - 1].relation == 'HED':
            for k, arc in enumerate(arcs):
                if arc.relation in ['SBV', 'VOB'] and arc.head == arcs[i].head and (words[k] in pattern or words[k] in ['证据', '证明']):
                    return -1
        if words[i] in pattern and words[arcs[i].head - 1] not in key_word and words[arcs[i].head - 1] in negative_word_list:
            return -1
    return 1


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
            factor_keywords = suqiu_keyword_dict[problem+'_'+suqiu]
        else:
            factor_keywords = problem_keyword_dict[problem]
        for i, line in enumerate(lines_training):
            match_result = single_case_match(line, problem, suqiu)
            X.append([match_result[f][1] if f in match_result else 0 for f in factor_keywords.index])
        np.save(target_file, X)
    except Exception:
        traceback.print_exc()


poses = {'n', 'v', 'p', 'd', 'nh', 'q', 'a', 'b', 'wp'}


def pos_filter(content):
    words = list(segmentor.segment(content))
    postags = list(postagger.postag(words))
    content = ''.join([w for i, w in enumerate(words) if postags[i] in poses])
    content = re.sub('[，。；：][，。；：]', '，', content)
    if content[0] in ['，', '。', '；', '：']:
        content = content[1:]
    return content


if __name__=='__main__':
    #print(single_case_match('1985年，原长寿县计划委员会以（85）6号文件，批准设立了长寿县第二水泥厂，工商行政管理机关登记的企业性质为集体企业（乡办），设立时的负责人为杨俊中。1998年7月30日，长寿县第二水泥厂申请企业注销，注销理由为整体出卖注销（购买人江涛），注销时的主管部门为长寿县葛兰镇乡镇企业办公室、经济性质股份合作制、法定代表人游中美。同年8月27日，工商行政管理机关批准长寿县第二水泥厂注销登记。《企业申请注销登记注册书》中记载，“企业人员全部由现有负责人江涛安排”。1998年7月15日，江涛申请登记设立了长寿县第二水泥厂（系个人独资企业，后更名为重庆市长寿区第二水泥厂）。2005年4月11日，江涛申请注销长寿县第二水泥厂，同日，工商行政管理机关批准重庆市长寿区第二水泥厂注销登记。注销登记档案中，重庆金盘山水泥有限公司、重庆润江水泥有限公司分别出具承诺，承诺内容分别为“重庆市长寿区第二水泥厂的全部债权债务由重庆金盘山水泥有限公司承担”、“重庆润江水泥有限公司愿意对重庆长寿区第二水泥厂的债权债务负联带责任”。2003年2月28日，由江涛等五个股东共同出资设立重庆金盘山水泥有限公司。2003年3月31日，重庆金盘山水泥有限公司经工商行政管理机关核准登记成立，公司类型有限责任公司，其中江涛占股份90%。2013年11月19日，重庆金盘山水泥有限公司的名称变更为重庆金盘山建材有限公司。2012年12月10日，原告梁小岗为乙方，被告金盘山公司为甲方，双方签订了《终止劳动关系协议》，协议载明：乙方于1998年8月进入甲方工作。2011年12月调动到重庆润江环保建材股份有限公司工作。在甲方工作年限13年4个月。经协商一致，按调动到重庆润江环保建材股份有限公司前十二个月的平均工资，甲方支付乙方经济补偿金38284元。乙方领取经济补偿金后，双方劳动关系终止。乙方不得以任何理由向甲方索要任何经济利益。事后，被告金盘山公司按该协议约定向原告支付经济补偿金38284元。2013年11月25日，原告梁小岗向重庆市长寿区劳动人事争议仲裁委员会申请仲裁，要求金盘山公司支付其加班工资、退还已扣取的失业保险金、赔偿欠缴的失业保险金、赔偿欠缴的养老保险金、补缴医疗保险金、退还已扣的生产安全保险金。同年12月3日，该仲裁委员会作出渝长劳人仲不字（2014）第2号不予受理案件通知书，以其请求超过劳动争议仲裁时效为由，决定不予受理。原告梁小岗不服该决定，就除退还已扣取的失业保险金、退还已扣的生产安全保险金外的其他仲裁请求，向本院分别提起诉讼，本院就原告梁小岗起诉要求被告金盘山公司支付1998年8月至2009年12月期间的养老保险赔偿金44903元以（2014）长法民初字第00067号案件立案受理。后该案在本院审理过程中，原告梁小岗于2014年10月24日向本院申请撤回起诉，本院予以准许。2015年6月23日，原告梁小岗再次向重庆市长寿区劳动人事争议仲裁委员会申请仲裁，要求被告金盘山公司向其支付因1998年1月1日至2009年12月31日期间未缴纳养老保险导致不能按月领取养老金的损失赔偿金46106元。该委于2015年6月23日作出渝长劳人仲不字〔2015〕第317号不予受理案件通知书，以原告梁小岗的仲裁请求超过仲裁时效为由，对原告梁小岗的仲裁请求不予受理。原告梁小岗不服，遂起诉来院。', '劳动社保', None))
    sentence='双方不能共同生活'
    print(single_case_match(sentence, '婚姻家庭', None))

    # words = list(segmentor.segment('不到两年'))  # 分词 元芳你怎么看
    # postags = list(postagger.postag(words))  # 词性标注
    # arcs = parser.parse(words, postags)  # 句法分析
    # print(words)
    # print(postags)
    # print([arc.head for arc in arcs])
    # print([arc.relation for arc in arcs])
