# -*- coding: utf-8 -*-
from config_loader import *
from data_util import text_underline


class Node(object):
    def __init__(self, index, name, type, question=None, answer=None):
        self.index = index  # 唯一索引
        self.name = name  # 节点名称
        self.type = type  # 节点类型，取值['root', 'law', 'situation', 'simple_result', 'conclusion', 'logic', 'factor']
        self.question = question  # 节点对应问题，只有factor节点有问题
        self.answer = answer  # 节点对应答案，只有factor节点有答案
        self.father = None  # 节点对应父节点
        self.sons = []  # 节点对应子节点

    def set_father(self, node):
        self.father = node

    def add_son(self, node):
        self.sons.append(node)

    def set_question(self, question):
        self.question = question

    def set_answer(self, answer):
        self.answer = answer


class LogicTree(object):
    def __init__(self, problem, suqiu, debug=True):
        # 读取配置文件
        self.problem = problem
        self.suqiu = suqiu
        self.debug = debug
        self.suqiu_advice = logic_ps_advice[problem+'_'+suqiu]
        self.suqiu_proof = logic_ps_proof[problem+'_'+suqiu]
        self.logic = logic_dict[problem + '_' + suqiu]

        self.question_factor = question_factor_dict[problem + '_' + suqiu]
        self.question_answer = question_answer_dict[problem + '_' + suqiu]
        self.question_father = question_father_dict[problem + '_' + suqiu]

        self.factor_question = factor_question_dict[problem + '_' + suqiu]
        self.factor_answer = factor_answer_dict[problem + '_' + suqiu]
        self.factor_unimportant = factor_unimportant_dict[problem + '_' + suqiu]
        self.factor_weight = factor_weight_dict[problem + '_' + suqiu]

        self.factor_pre = factor_pre_dict[problem + '_' + suqiu]
        self.factor_son = factor_son_dict[problem + '_' + suqiu]
        self.factor_exclusion = factor_exclusion_dict[problem + '_' + suqiu]

        self.factor_proof = factor_proof_dict[problem + '_' + suqiu]
        self.factor_replace = factor_replace_dict[problem + '_' + suqiu]

        self.candidate_question = candidate_question_dict[problem + '_' + suqiu]
        self.candidate_factor = candidate_factor_dict[problem + '_' + suqiu]

        # 初始化树
        self.root = None  # 根节点
        self.count = 0  # 节点数目
        self.match_flag = False  # 是否有特征正向匹配到
        self.last_question = None  # 上一个提问问题
        self.logic_result = None  # 支持结果
        self.support_proof = None  # 支持证据
        self.factor_nodes = {}  # 所有特征节点
        self.factor_flag = {}  # 特征匹配情况。flag取值[0,1,-1]，0未匹配到，1正向匹配，-1负向匹配
        self.factor_sentence = {}  # 特征匹配的句子。flag取值[0,1,-1]，0未匹配到，1正向匹配，-1负向匹配

        self.load_nodes()
        self.paths = self.get_all_support_path()

    def _get_node_type(self, name, parent_node):
        """
        获取当前节点的类型
        :param parent_node:
        :return:
        """
        if parent_node is None:
            return 'root'
        elif parent_node.type == 'root':
            return 'law'
        elif parent_node.type == 'law':
            return 'situation'
        elif parent_node.type == 'situation':
            return 'simple_result'
        elif parent_node.type == 'simple_result':
            return 'conclusion'
        elif parent_node.type == 'conclusion':
            return 'advice'
        elif name in ['and', 'or']:
            return 'logic'
        else:
            return 'factor'

    def _load_nodes(self, node_dict, parent_node):
        """
        读取当前节点信息构造节点，并判断类型，如果是特征节点，则加到nodes中
        递归添加其子节点
        :param node_dict: 字典，包含当前节点及其子节点信息
        :param parent_node: 当前节点的父节点
        :return:
        """
        name = node_dict['title'].strip()
        type = self._get_node_type(name, parent_node)
        node = Node(self.count, name, type)
        self.count += 1
        if type == 'factor':
            if name in self.factor_nodes:
                self.factor_nodes[name].append(node)
            else:
                self.factor_nodes[name] = [node]
            if name not in self.factor_flag:
                self.factor_flag[name] = 0
                self.factor_sentence[name] = None
            if name in self.factor_son:
                for f in self.factor_son[name]:
                    if f not in self.factor_flag:
                        self.factor_flag[f] = 0
                        self.factor_sentence[f] = None
        if type == 'root':
            self.root = node
        else:
            node.set_father(parent_node)
            parent_node.add_son(node)
        if 'topics' in node_dict:
            for nd in node_dict['topics']:
                self._load_nodes(nd, node)

    def _set_question_answer(self):
        """
        设置节点对应的问答
        :return:
        """
        for name, nodes in self.factor_nodes.items():
            if name in self.factor_question:
                for node in nodes:
                    node.set_question(self.factor_question[name])
                    node.set_answer(self.factor_answer[name])

    def load_nodes(self):
        """
        加载配置文件，并添加问答信息
        :return:
        """
        self._load_nodes(self.logic, None)
        self._set_question_answer()

    def restart(self):
        """
        重置匹配信息
        :return:
        """
        for factor in self.factor_flag:
            self.factor_flag[factor] = 0
            self.factor_sentence[factor] = None

    def _print_node(self, depth, node):
        """
        打印当前节点信息
        :param depth: 当前节点的深度
        :param node: 当前节点
        :return:
        """
        print('\t' * depth, node.index, node.name, node.type)
        for son_node in node.sons:
            self._print_node(depth + 1, son_node)

    def print_tree(self):
        """
        递归打印树信息
        :return:
        """
        self._print_node(0, self.root)

    def print_factor(self):
        """
        打印所有需要匹配的特征
        :return:
        """
        for name in self.factor_nodes.keys():
            print(name, self.factor_flag[name], self.factor_sentence[name])

    def _get_support_path(self, node, path, result):
        """
        根据当前节点的类型进行相应操作。
        root, law, situation, simple_result, logic and: 不做任何操作，进行子节点递归
        conclusion: 将节点添加到path中，进行子节点递归
        support: 新建路径，并在递归完成时将路径添加到result中
        logic or: 根据节点数量复制多份path，进行子节点递归
        factor: 将当前节点添加到path中
        :param node: 当前节点
        :param path: 当前节点之前的路径
        :param result: 保存已经走通的路径
        :return:
        """
        name = node.name
        type = node.type
        if type == 'advice':
            path = [[node]]
            for son_node in node.sons:
                self._get_support_path(son_node, path, result)
            result += path
        elif type == 'logic' and name == 'or':
            temp = [[p for p in pt] for pt in path]
            first = True
            for son_node in node.sons:
                if first:
                    self._get_support_path(son_node, path, result)
                    first = False
                else:
                    t = [[p for p in pt] for pt in temp]
                    self._get_support_path(son_node, t, result)
                    path += t
        elif type == 'factor':
            for p in path:
                p.append(node)
        else:
            for son_node in node.sons:
                self._get_support_path(son_node, path, result)

    def get_all_support_path(self):
        """
        获取所有连通路径
        :return:
        """
        result = []
        path = []
        self._get_support_path(self.root, path, result)
        return result

    def print_paths(self):
        """
        打印所有连通路径
        :return:
        """
        for path in self.paths:
            p = ''
            for node in path[1:]:
                p += str(node.index) + node.name + '+'
            p = p[:-1] + '-->\n' + path[0].father.father.name + '\n' + path[0].father.name + '\n' + path[0].name
            print(p)

    def export_paths(self):
        for path in self.paths:
            p = ''
            for i in range(1, 11):
                if i < len(path):
                    p += path[i].name + '\t'
                else:
                    p += '\t'
            p = p + path[0].father.father.name + '\t' + path[0].father.name.replace('\n', ' ') + '\t' + path[0].name.replace('\n', ' ')
            print(p)

    def print_path(self, path):
        """
        打印所有连通路径
        :return:
        """
        p = ''
        for node in path[1:]:
            p += str(node.index) + node.name + '+'
        p = p[:-1] + '-->' + path[0].father.father.name
        print(p)

    def _set_match_flag(self, factor, flag):
        """
        判断是否有特征正向匹配到
        :param factor:
        :param flag:
        :return:
        """
        if flag != 1 or self.match_flag:
            return
        if factor in self.factor_unimportant:
            return
        for path in self.paths:
            if path[0].father.father.name in ['前提', '大前提', '不支持']:
                continue
            for node in path[1:]:
                if node.name == factor:
                    self.match_flag = True
                    return

    def _set_factor_flag(self, factor, flag, sentence):
        """
        设置特征的flag，并对子特征也进行设置
        :param factor:
        :param flag:
        :return:
        """
        if factor in self.factor_flag:
            self.factor_flag[factor] = flag
            self.factor_sentence[factor] = sentence
            self._set_match_flag(factor, flag)
            if flag == 1 and factor in self.factor_son:
                for f in self.factor_son[factor]:
                    if f not in self.factor_flag:
                        continue
                    if self.factor_flag[f] == 1:
                        continue
                    self._set_factor_flag(f, flag, sentence)

    # def _get_factor_description(self, factor, flag):
    #     """
    #     获取特征对应的描述，用于将问题答案转化为描述，提供给模型预测
    #     :param factor:
    #     :param flag:
    #     :return:
    #     """
    #     if flag==1 and factor in self.factor_positive:
    #         return self.factor_positive[factor]
    #     elif flag==-1 and factor in self.factor_negative:
    #         return self.factor_negative[factor]
    #     return ''

    def add_match_result(self, factor, flag, sentence):
        """
        将匹配结果添加到树的factor_flag中。需要判断匹配的节点是否在树中。
        对于特征之间的互斥关系，将匹配结果是1或者互斥特征数量是2的互斥节点也更新为-flag
        :param factor: 匹配到的特征
        :param flag: 1或-1，正向或负向匹配
        :return: bool，是否添加信息到树中
        """
        self._set_factor_flag(factor, flag, sentence)
        if factor not in self.factor_exclusion:
            return
        if flag == 1 or len(self.factor_exclusion[factor]) == 1:
            for f in self.factor_exclusion[factor]:
                self._set_factor_flag(f, -flag, sentence)

    # def get_candidate_question(self):
    #     """
    #     在什么都没匹配到的情况下选择候选问题进行提问
    #     :return:
    #     """
    #     precondition_path, paths = self._filter_path_by_precondition(self.paths)
    #     if len(paths)==0:
    #         self.logic_result = self.get_precondition_result(precondition_path[0])
    #         return None
    #     return self.candidate_question
    #
    # def add_candidate_question_result(self, answers):
    #     """
    #     将候选问题的回答添加到树中, 如果未匹配到特征, 直接返回结果
    #     :param answers:
    #     :return:
    #     """
    #     for factor, answer in self.candidate_factor.items():
    #         if answer in answers:
    #             self._set_factor_flag(factor, 1)
    #         else:
    #             self._set_factor_flag(factor, -1)
    #     if self.match_flag:
    #         return self.get_next_question()
    #     else:
    #         self.logic_result = ['不支持', '你的情形与法律支持的离婚情形不太相符，很难得到支持。']
    #         return None

    def add_question_result(self, question, answers):
        """
        将问答结果添加到树的factor_flag中。更新相同问题的所有特征，特征答案在选择答案中的flag置为1，否则置为-1
        :param question: 提问的问题
        :param answers: 用户选择的答案
        :return:
        """
        factors = []
        if question in self.question_father:
            question = self.question_father[question]
        # print('add_question_result', question, answers)
        if question == self.candidate_question:
            for answer in answers:
                if answer not in self.candidate_question.split(':')[1].split(';'):
                    raise Exception(answer + '不在答案列表中')
            for factor, answer in self.candidate_factor.items():
                add = False
                for ans in answer.split('|'):
                    if ans in answers:
                        factors.append(factor)
                        add = True
                        self._set_factor_flag(factor, 1, question + '_' + ans)
                        # if factor in self.factor_exclusion:
                        #     for f in self.factor_exclusion[factor]:
                        #         self._set_factor_flag(f, -1, question + ':' + ans)
                        break
                if not add:
                    self._set_factor_flag(factor, -1, None)
            self.match_flag = True
            return factors
        if question not in self.question_answer:
            return factors
        if question not in self.question_factor:
            return factors
        for answer in answers:
            if answer not in self.question_answer[question]:
                raise Exception(answer + '不在答案列表中：%s' % (','.join(self.question_answer[question])))
        self.last_question = question
        for factor in self.question_factor[question]:
            add = False
            for answer in self.factor_answer[factor].split('|'):
                if answer in answers:
                    factors.append(factor)
                    self._set_factor_flag(factor, 1, question + '_' + answer)
                    add = True
                    # if factor in self.factor_exclusion:
                    #     for f in self.factor_exclusion[factor]:
                    #         self._set_factor_flag(f, -1, question + ':' + answer)
                    break
            if not add:
                self._set_factor_flag(factor, -1, None)
                if self.factor_answer[factor] == '是':
                    factors.append(question.split(':')[0].replace('是否', '没有').replace('？',''))
                elif self.factor_answer[factor] == '否':
                    factors.append(question.split(':')[0].replace('是否', '').replace('？',''))
                elif self.factor_answer[factor] == '有':
                    factors.append(question.split(':')[0].replace('有没有', '没有').replace('？',''))
                elif self.factor_answer[factor] == '没有':
                    factors.append(question.split(':')[0].replace('有没有', '有').replace('？',''))
        return factors

    def _filter_path_by_precondition(self, paths):
        """
        遍历不满足的前提路径，并过滤掉相关路径
        :return: [不满足的前提路径, 剩余候选路径]
        """

        # 统计每个前提有几条支持路径
        preconditions = {}
        for path in paths:
            if path[0].father.father.name not in ['前提', '大前提']:
                continue
            preconditions[path[0].index] = preconditions.get(path[0].index, 0) + 1

        # 遍历查找所有不满足条件的前提路径
        prediction_path = []
        for path in paths:
            if path[0].father.father.name not in ['前提', '大前提']:
                continue
            for node in path[1:]:
                if self.factor_flag[node.name] == -1:
                    preconditions[path[0].index] -= 1
                if preconditions[path[0].index] == 0:
                    if path[0].father.father.name == '大前提':
                        return [path], []
                    prediction_path.append(path)
                    break

        # 过滤掉所有不满足条件的前提路径所关联的路径
        if len(prediction_path) > 0:
            indices = []
            for path in prediction_path:
                for n1 in path[0].father.father.father.father.sons:
                    for n2 in n1.sons:
                        for n3 in n2.sons:
                            for n4 in n3.sons:
                                indices.append(n4.index)
            paths = [path for path in paths if path[0].index not in indices and path[0].father.father.name not in ['前提', '大前提']]
            return prediction_path, paths

        paths = [path for path in paths if path[0].father.father.name not in ['前提', '大前提']]
        # 默认返回所有支持路径
        return [], paths

    def get_next_question(self):
        """
        获取下一个问题。
        过滤掉所有有特征为-1的路径。如果单个prediction的所有路径均为-1，则返回不满足前提
        过滤掉所有特征均为0的路径。对于未匹配到信息的路径，认为和用户情形无关，不去提问
        计算剩余路径中每个特征的最少问答次数和总共出现次数，优先问问答次数最少的，问答次数相同问总共出现次数最高的
        如果没有剩余路径，则表示用户情形不满足支持条件，返回不支持结果
        :return: [问题, 结果]
        """
        # 过滤不满足前提的路径
        precondition_path, paths = self._filter_path_by_precondition(self.paths)

        # 遍历每条路径
        questions = {}
        unsupport_question = None
        father_question = None
        support_path = None
        unsupport_path = None
        for path in paths:
            if_pass = False  # 是否跳过，有节点为-1则跳过，或者有节点没有对应问题
            if_ask = False  # 是否提问，有节点为1则提问
            temp_unmatch = []  # 所有待提问的特征节点和对应问题
            temp_match = []
            for node in path[1:]:
                if self.factor_flag[node.name] == -1:
                    if_pass = True
                    break
                if self.factor_flag[node.name] == 1:
                    if_ask = True
                    temp_match.append(node)
                if self.factor_flag[node.name] == 0:
                    if node.question is None:
                        if_pass = True
                        break
                    temp_unmatch.append(node)
            if if_pass:
                continue
            if len(temp_unmatch) == 0:
                if path[0].father.father.name == '不支持':
                    unsupport_path = path
                else:
                    support_path = path
                continue
            if if_ask:
                if self.debug:
                    self.print_path(path)
                for node in temp_unmatch:
                    # 如果节点存在父节点，则判断父节点flag是否为1，为1直接提问该节点的问题
                    if node.name in self.factor_pre:
                        for f in self.factor_pre[node.name]:
                            if self.factor_flag[f] == 1:
                                father_question = node.question.replace('[FF]', f)
                        continue
                    if path[0].father.father.name == '不支持':
                        unsupport_question = node.question
                    weight = self.factor_weight[node.name] if node.name in self.factor_weight else 0
                    if node.question in questions:
                        questions[node.question] = [min(len(temp_unmatch), questions[node.question][0]),
                                                    max(len(temp_match), questions[node.question][1]),
                                                    max(weight, questions[node.question][2])]
                    else:
                        questions[node.question] = [len(temp_unmatch), len(temp_match), weight]

        # 大前提不满足返回结果
        if len(paths) == 0 and len(precondition_path) > 0:
            self.logic_result = self.get_precondition_result(precondition_path[0])
            return None

        # 有不支持路径连通返回结果
        if unsupport_path is not None:
            self.logic_result = self.get_support_result(unsupport_path, self.debug)
            return None

        # 提问不支持路径的问题
        if unsupport_question is not None:
            return unsupport_question

        # 有路径连通返回结果
        if support_path is not None:
            self.logic_result = self.get_support_result(support_path, self.debug)
            self.support_proof = self.get_support_proof(support_path)
            return None

        # 提问有父特征的问题
        if father_question is not None:
            return father_question

        # 有问题要提问，则按照最短路径和出现次数进行排序提问
        if len(questions) > 0:
            question = sorted(questions.items(), key=lambda x: x[1][0] * 50000 - x[1][1] * 5000 - x[1][2]/100)[0][0]
            return question

        # 未匹配到特征提问候选问题
        if not self.match_flag:
            return self.candidate_question

        # 没有问题要提问，判断是否有前提未满足
        if len(precondition_path) > 0:
            self.logic_result = self.get_precondition_result(precondition_path[0])
            return None

        # 根据上一个问题获取结果
        self.logic_result = self.get_unsupport_result(paths)
        return None

    def get_precondition_result(self, path):
        """
        返回前提路径对应的结果
        :param path:
        :return:
        """
        return ['不满足前提', path[0].father.name, self.suqiu_advice]

    def get_support_result(self, path, debug):
        """
        返回支持路径对应的结果
        :param path:
        :return:
        """
        result = path[0].father.name
        if '[FF]' in result:
            for p in path[1:]:
                if p.name in self.factor_replace:
                    result = result.replace('[FF]', p.name)
        advice = path[0].name
        if len(advice)<10:
            advice = self.suqiu_advice
        if debug:
            return [path[0].father.father.name, '+'.join(p.name for p in path[1:]) + '-->\n' + result, advice]
        else:
            return [path[0].father.father.name, result, advice]

    def get_support_proof(self, path):
        """
        获取路径相关的证据
        :param path:
        :return:
        """
        proof = '根据您所描述的事实，建议搜集以下能够证明您所描述事实的证据：\n'
        count = 0
        for p in path[1:]:
            factor = p.name.replace('_', '')
            if factor in self.factor_proof:
                if '用以证明' in self.factor_proof[factor]:
                    proof += '%d. %s；\n' % (count + 1, self.factor_proof[factor])
                else:
                    proof += '%d. %s等，用以证明%s的事实；\n' % (count + 1, self.factor_proof[factor], factor)
                count += 1
        proof = proof[:-2] + '。'
        if len(self.suqiu_proof)>0:
            proof += '\n' + self.suqiu_proof
        return proof

    def get_unsupport_result(self, paths):
        """
        返回不支持路径对应的结果
        :param path:
        :return:
        """
        if self.last_question is None:
            return ['不支持', '您的情形与法律支持情形匹配度较低，可能无法得到支持。', self.suqiu_advice]

        for path in paths:
            if path[0].father.father.name != '支持':
                continue
            for node in path[1:]:
                if node.name in self.question_factor[self.last_question]:
                    factors1 = [p.name for p in path[1:] if self.factor_flag[p.name] == 1]
                    factors2 = [p.name for p in path[1:] if self.factor_flag[p.name] == -1]
                    if len(factors1) == 0 or len(factors2) == 0:
                        continue
                    return ['不支持', '根据您的描述，虽然满足【%s】，但由于不满足【%s】的条件，理论上您的诉求支持率不高。' % (factors1[0], factors2[0]), self.suqiu_advice]
        return ['不支持', '缺少关键信息，无法给出准确评估。', self.suqiu_advice]

    def get_logic_result(self):
        support = self.logic_result[0]
        if self.logic_result[0] == '不满足前提':
            result = -1
            reason = self.logic_result[1]
            proof = ''
            advice = re.sub('^法律建议[，：:,]{0,1}', '', self.logic_result[2])
            if advice.startswith('\n'):
                advice = advice[len('\n'):]
        elif self.logic_result[0] == '支持':
            result = 1
            reason = self.logic_result[1]
            if len(self.support_proof)>0:
                proof = self.support_proof
            else:
                proof = ''
            advice = re.sub('^法律建议[，：:,]{0,1}', '', self.logic_result[2])
            if advice.startswith('\n'):
                advice = advice[len('\n'):]
        else:
            result = -1
            reason = self.logic_result[1]
            proof = ''
            advice = re.sub('^法律建议[，：:,]{0,1}', '', self.logic_result[2])
            if advice.startswith('\n'):
                advice = advice[len('\n'):]
        if not self.debug:
            reason = text_underline(reason)
            proof = text_underline(proof)
            advice = text_underline(advice)
        if self.problem=='婚姻继承' and self.suqiu=='房产分割' and len(re.findall('属于.{0,5}个人财产', reason))>0 and '夫妻共同财产' not in reason:
            result = -1
            support = '不支持'
        return result, reason, proof, advice, support

    def print_logic_result(self):
        """
        打印评估理由
        :return:
        """
        result, reason, proof, advice, support = self.get_logic_result()
        print(support)
        print('理由: %s' % (reason))
        print('证据: %s' % (proof))
        print('法律建议: %s' % (advice))
        return result


if __name__=='__main__':
    tree = LogicTree('交通事故', '赔偿主体')
    tree.export_paths()
