# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.abspath('../../../'))
sys.path.append(os.path.abspath('../common'))
from config_loader import *
from logic_tree import LogicTree


########################################################################################################################
#
# 配置表检查
#
########################################################################################################################

for problem, suqius in logic_ps.items():

    # 纠纷类型句式格式检查
    data = pd.read_csv(config_path + problem + '/' + problem + '特征.csv', encoding='utf-8')
    for index, row in data.iterrows():
        factor = row['factor']
        sentence = row['sentences']
        for sents in sentence.split('|'):
            for sent in re.findall('【([^【】]*)】', sents):
                if '[' not in sent:
                    if sent not in sentence_keyword_dict:
                        print('纠纷类型句式格式检查错误：', problem, factor, sentence)
                else:
                    for st in re.findall('\[([^\[\]]*)\]', sent):
                        if st not in sentence_keyword_dict:
                            print('纠纷类型句式格式检查错误：', problem, factor, sentence)

    for suqiu in suqius:
        print('*'*20 + problem + ', ' + suqiu + '*'*20)
        # 构造逻辑树
        tree = LogicTree(problem, suqiu)

        # 每棵树的非前提特征
        factors = []
        for path in tree.paths:
            if path[0].father.father.name in ['前提', '大前提']:
                continue
            for node in path[1:]:
                factors.append(node.name)

        # 候选问答和普通问答一致性检查
        if candidate_question_dict[problem + '_' + suqiu] not in factor_question_dict[problem + '_' + suqiu].values:
            print(candidate_question_dict[problem + '_' + suqiu] + ' 不在普通问答里')
        else:
            for factor, answer in candidate_factor_dict[problem + '_' + suqiu].items():
                if answer != factor_answer_dict[problem + '_' + suqiu][factor]:
                    print(factor, answer, ' 候选问答答案和普通问答不一致')

        # 检查是否有路径没有对应证据
        for path in tree.paths:
            if path[0].father.father.name!='支持':
                continue
            has_proof = False
            for node in path[1:]:
                if node.name in tree.factor_proof:
                    has_proof = True
            if not has_proof:
                print('+'.join(p.name for p in path[1:]) + '-->' + '该路径没有找到对应证据')

        # 检查是否有路径[FF]没有替换
        for path in tree.paths:
            if '[FF]' not in path[0].father.name:
                continue
            has_replace = False
            for node in path[1:]:
                if node.name in tree.factor_replace:
                    has_replace = True
            if not has_replace:
                print('+'.join(p.name for p in path[1:]) + '-->' + '该路径没有找到替换特征')

        # 检查是否有节点没有对应问答
        for name, nodes in tree.factor_nodes.items():
            if name not in tree.factor_question and name in factors:
                print(name, ' 节点没有对应问答')

        # 检查是否有节点没有对应关键词
        for name, nodes in tree.factor_nodes.items():
            if name not in suqiu_bkw_dict[problem+'_'+suqiu]:
                print(name, ' 节点没有对应关键词')

        # 检查候选特征是否在诉求特征里
        for factor, answer in tree.candidate_factor.items():
            if factor not in tree.factor_flag:
                print(factor, ' 在候选特征但是不在诉求特征里')

        # 检查特征答案是否在问题答案里
        for factor, answer in tree.factor_answer.items():
            answers = tree.factor_question[factor].split(':')[1].split(';')
            for ans in answer.split('|'):
                if ans not in answers:
                    print(factor, answer, ' 特征答案不在问题答案里')

        # 检查是否有诉求节点
        for name in tree.factor_nodes.keys():
            if name.startswith('诉求'):
                print(name, ' 为诉求节点')
