# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from vectorizer import *
from similarity import *

"""
对规则文件进行文本解析，输出节点和逻辑结构的候选文件
"""


class Analyzer(object):
    """文本解析器接口定义"""
    def __init__(self,
                 task_config_path,
                 logic_config_path,
                 node_config_path,
                 keyword_config_path):
        self.task_config_path = task_config_path
        self.logic_config_path = logic_config_path
        self.node_config_path = node_config_path
        self.keyword_config_path = keyword_config_path
        self.logic_config_cand_path = self._get_cand_path(logic_config_path)
        self.node_config_cand_path = self._get_cand_path(node_config_path)
        self.keyword_config_cand_path = self._get_cand_path(keyword_config_path)

    def _get_cand_path(self, path):
        name = path[:path.rindex('.')]
        type = path[path.rindex('.')+1:]
        return name+'_cand'+type

    def read_context(self, tc_path):
        """
        从任务配置中读取每个任务对应的文档内容
        :return: {任务id: 任务pattern}, {任务id: 任务context}
        """
        raise NotImplementedError()

    def split_or_condition(self, condition):
        """
        按照or类型的判断规则将提取的条件分成多个并列子条件. 需要重写.
        :param condition: 条件字符串
        :return: [子条件1, 子条件2, ...]
        """
        raise NotImplementedError()

    def split_and_condition(self, condition):
        """
        按照and类型的判断规则将提取的条件分成多个子条件组合. 需要重写.
        :param condition: 条件字符串
        :return: [子条件1, 子条件2, ...]
        """
        raise NotImplementedError()

    def extract_logic_structure(self, pattern, context):
        """
        逻辑规则的提取方法
        :return: [(条件, 结论)]
        """
        raise NotImplementedError()

    def check_similar_nodes(self, nodes, threshold=0.9):
        """
        检查哪些节点相似度较高，可以进行合并
        """
        vectorizer = Word2VecVectorizer()
        vectors = [vectorizer.vectorize(node) for node in nodes]
        for i in range(len(nodes)):
            if vectors[i] is None:
                continue
            for j in range(i, len(nodes)):
                if vectors[j] is None:
                    continue
                sim = cosine(vectors[i], vectors[j])
                if sim>threshold and nodes[i]!=nodes[j]:
                    print('%s and %s are similar nodes' % (nodes[i], nodes[j]))

    def run(self):
        """
        任务主函数
        1. 若逻辑配置文件不存在，则对规则文件进行解析，生成候选逻辑配置文件, 用户人工修改后改为逻辑配置文件
        2. 从逻辑配置文件中读取节点信息，进行相似度计算，并提示合并
        3. 节点配置文件候选和逻辑配置文件候选
        """
        task_pattern, task_context = self.read_context(self.task_config_path)
        if not os.path.exists(self.logic_config_path):
            data = []
            for taskid, pattern in task_pattern.items():
                logic = self.extract_logic_structure(pattern, task_context[taskid])
                for l in logic:
                    for c in self.split_or_condition(l[0]):
                        data.append([taskid, '+'.join(self.split_and_condition(c)), l[1]])
            data = pd.DataFrame(data, columns=['id', 'condition', 'conclusion'])
            data.to_csv(self.logic_config_cand_path, index=False, encoding='utf-8')
            return

        data = pd.read_csv(self.logic_config_path, encoding='utf-8')
        nodes = []
        for v in data['condition'].values:
            nodes += v.split('+')
        self.check_similar_nodes(nodes)


class BasicAnalyzer(Analyzer):
    """文本基础解析器"""

    def __init__(self,
                 task_config_path,
                 logic_config_path,
                 node_config_path,
                 keyword_config_path):
        super().__init__(task_config_path,
                         logic_config_path,
                         node_config_path,
                         keyword_config_path)

        self.or_pattern = ['\n', '[;；]', '（[一二三四五六七八九十\d]+）', '[一二三四五六七八九十\d]+[、\.，,]']

    def read_context(self, tc_path):
        task_config = pd.read_csv(tc_path, encoding='utf-8')
        task_config = task_config[task_config['status']==1]

        task_pattern = task_config['pattern'].groupby(task_config['id']).agg(lambda x: list(x)[0])

        task_context = {}
        for index, row in task_config.iterrows():
            with open(os.path.join('../data', row['doc_path']), 'r', encoding='utf-8') as file:
                task_context[row['id']] = ''.join(file.readlines())

        return task_pattern, task_context

    def split_or_condition(self, condition):
        result = []
        for pattern in self.or_pattern:
            if len(re.findall(pattern, condition))>0:
                for r in re.split(pattern, condition):
                    if len(r.strip())>0:
                        result += self.split_or_condition(r)
                return result
        if len(re.findall('，以及|，或者|以及|或者|或|及', condition))>0:
            temp_rs = re.split('，以及|，或者|以及|或者|或|及', condition)
            start = temp_rs[0]
            end = temp_rs[-1]
            find_start = False
            for r in temp_rs:
                for i in range(4,0,-1):
                    if r[:i] in start:
                        if r!=start:
                            find_start = True
                        r = start[:start.index(r[:i])] + r
                        break
                for i in range(4,0,-1):
                    if r[-i:] in end:
                        r = r + end[end.index(r[-i:]) + i:]
                        break
                result += self.split_or_condition(r)
            if not find_start:
                return [condition]
            return result
        else:
            return [condition]

    def split_and_condition(self, condition):
        return [condition]

    def extract_logic_structure(self, pattern, context):
        """
        逻辑规则的提取方法
        :return: [(条件, 结论)]
        """
        logic = []
        for p in pattern.split('|'):
            if p.index('【结论】') < p.index('【条件】'):
                condition_index = 1
                conclusion_index = 0
            else:
                condition_index = 0
                conclusion_index = 1

            result = re.findall(p.replace('【结论】', '([\s\S]*?)').replace('【条件】', '([\s\S]*?)'), context)
            for r in result:
                logic.append((r[condition_index], r[conclusion_index]))
        return logic


if __name__=='__main__':
    config_path = '../config'
    tc_path = os.path.join(config_path, 'task.csv')
    lc_path = os.path.join(config_path, 'logic.csv')
    nc_path = os.path.join(config_path, 'node.csv')
    kc_path = os.path.join(config_path, 'keyword.csv')

    analyzer = BasicAnalyzer(tc_path, lc_path, nc_path, kc_path)
    analyzer.run()
