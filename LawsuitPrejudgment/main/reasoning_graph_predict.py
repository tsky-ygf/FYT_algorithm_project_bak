# -*- coding: utf-8 -*-
import logging
import numpy as np
import collections

from LawsuitPrejudgment.common.config_loader import *
from LawsuitPrejudgment.common import single_case_match, LogicTree, get_next_suqiu_or_factor
from LawsuitPrejudgment.common.data_util import text_underline
<<<<<<< Updated upstream
from LawsuitPrejudgment.lawsuit_prejudgment.constants import FEATURE_TOGGLES_CONFIG_PATH, \
    HTTP_SITUATION_CLASSIFIER_SUPPORT_PROBLEMS_CONFIG_PATH
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.civil_report_action import CivilReportAction
from LawsuitPrejudgment.lawsuit_prejudgment.core.actions.civil_report_action_message import CivilReportActionMessage
from LawsuitPrejudgment.lawsuit_prejudgment.feature_toggles import FeatureToggles
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.http_based_situation_classifier import HttpClient, \
    HttpBasedSituationClassifier, DataTransferObject
from LawsuitPrejudgment.lawsuit_prejudgment.nlu.situation_classifiers.situation_classifier_message import \
    SituationClassifierMessage
from Utils.io import read_json_attribute_value
=======
>>>>>>> Stashed changes
from LawsuitPrejudgment.prediction.bert_predict import predict as predict

logging.basicConfig(level=logging.DEBUG)

match_factor_group = {
    '劳动社保': [['养老保险', '医疗保险', '失业保险', '生育保险']],
    '借贷纠纷': [['金融借贷', '个人之间借贷', '企业之间借贷', '个人与企业之间借贷']],
    # "租赁合同":[[]],
}


def predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list_, debug=False):
    feature_toggles = FeatureToggles(FEATURE_TOGGLES_CONFIG_PATH)
    http_classifier_support_problems = read_json_attribute_value(HTTP_SITUATION_CLASSIFIER_SUPPORT_PROBLEMS_CONFIG_PATH,
                                                                 "support_problems")

    if feature_toggles.http_situation_classifier.enabled and problem in http_classifier_support_problems:
        return _predict_by_http(problem, claim_list, fact)
    else:
        return _predict_by_factor(problem, claim_list, fact, question_answers, factor_sentence_list_, debug)


def _predict_by_http(problem, claim_list, fact):
    # TODO 多个诉求的处理
    claim = claim_list[0]

    # get situation by HttpBasedSituationClassifier
    situation_classifier = HttpBasedSituationClassifier(HttpClient(url="http://172.19.82.199:7998/situationreview"), DataTransferObject())
    resp_json = situation_classifier.classify_situations(SituationClassifierMessage(claim, fact))
    claim_from_http = resp_json.get("claim", "")
    situation = resp_json.get("situation", "")

    # get report by CivilReportAction
    action = CivilReportAction()
    message = CivilReportActionMessage(problem, claim_from_http, situation, fact)
    result = action.run(message)

    return result


def _predict_by_factor(problem, claim_list, fact, question_answers, factor_sentence_list_, debug=False):
    """
    推理图谱即评估新版本-预测的主入口。
    1. 加载逻辑树-->2.定义共现变量--->3.设置树的状态：做文本匹配、结合已经问过的问题和设置模型特征；问一个问题--->4.如果没有问题了，那么需要出评估报告
    :param problem: 纠纷类型
    :param claim_list: 诉求列表
    :param fact: 文本描述. e.g.fact=''婚后男⽅方⽗父⺟母出资⾸首付，夫妻名义贷款还贷，房产证只写男⽅方名，离婚后财产如何分配
    :param question_answers: 已经问过的问题和答案的字典(dict). e.g. question_answers={“付款⽅方式是以下哪种情形?:全款;首付”:”首付”,“您是婚前还是婚后买房?:婚前;婚后“:”婚后”}
    :param factor_sentence_list_:文本匹配到的特征
    :return: a dict, contain fields of question_asked, question_next,factor_sentence_list,report_dict(reason_of_evaluation,evidence_module,legal_advice,possibility_support,support_or_not)
    """
    #####################################################################################################################
    claim_list = ['减少租金或者不支付租金' if c == '减少租金或则不支付租金' else c for c in claim_list]

    # 1. 初始化诉求结果
    logging.info('5.1. initial suqiu result')
    logic_problem_suqius = []
    for suqiu in user_ps[problem]:
        if suqiu not in claim_list:
            continue
        logic_problem_suqius += user_ps2logic_ps[problem + '_' + suqiu]

    # Kiwi Debug
    logging.info('logic_problem_suqius:')
    logging.info(str(logic_problem_suqius))

    prob_problem_suqius = []
    for suqiu in user_ps[problem]:
        if suqiu not in claim_list:
            continue
        prob_problem_suqius += user_ps2prob_ps[problem + '_' + suqiu]

    # 2. 诉求选择对应的默认特征
    logging.info('5.2. add logic suqiu factor')
    suqiu_factor = {}
    for ps in logic_problem_suqius:
        if ps in logic_ps_factor:
            for f, v in logic_ps_factor[ps].items():  # 诉求配置中的logic_ps_factor
                suqiu_factor[f] = v

    # 3. 特征匹配
    logging.info('5.3. factor match')
    factor_sentence_list = {}
    if len(factor_sentence_list_) > 0 or len(question_answers) > 0:
        # KIWI:不是首轮，已经匹配过特征，直接取出来就行。
        for factor_flags in factor_sentence_list_:
            sentence_matched, factor, flag, _ = factor_flags
            factor_sentence_list[factor] = [sentence_matched, flag]
    else:
        # 为什么在所有的logic_problem_suqius中匹配特征？
        for problem in set([ps.split('_')[0] for ps in logic_problem_suqius]):
            sentence_factor_dict = single_case_match(fact, problem, None)  # 返回字典：{factor: [匹配的句子，0/1/-1]}
            for factor, sentence in sentence_factor_dict.items():
                sentence_matched, flag = sentence
                factor_sentence_list[factor] = [sentence_matched, flag]

                # KIWI:我理解是，如['金融借贷', '个人之间借贷','企业之间借贷','个人与企业之间借贷']是互斥的。
                # KIWI:正向匹配到其中一个特征，则设置其余特征为负向匹配。
                if flag == -1:
                    continue
                if problem not in match_factor_group:
                    continue
                # problem是劳动社保，借贷纠纷之一，特定factor全部设置为-1 ？
                for factor_group in match_factor_group[problem]:
                    if factor in factor_group:
                        for f in factor_group:
                            if f not in factor_sentence_list:
                                factor_sentence_list[f] = [sentence_matched, -1]

    # 4. 设置树的状态, 并提取下一个问题
    logging.info('5.4. create tree and get next question')
    question_next = None
    question_type = '1'
    question_answer_str = ''
    suqiu_tree = {}
    suqiu_result = {}
    # KIWI
    # logging.info('logic_ps_result:' + ' ' + str(logic_ps_result))
    debug_info = collections.OrderedDict()

    for ps in logic_problem_suqius:
        problem, suqiu = ps.split('_')
        # KIWI
        logging.info('当前诉求:' + ' ' + str(suqiu))
        suqiu_debug_info = None
        next_question_debug_info = None

        next = get_next_suqiu_or_factor(ps, suqiu_result, suqiu_factor, question_answers, factor_sentence_list)
        if next is None:  # 有前置诉求或特征并且不满足
            suqiu_result[suqiu] = -1
            if ps in logic_ps_result:
                tree = LogicTree(problem, suqiu, debug)
                tree.logic_result = ['不满足前提', logic_ps_result[ps], tree.suqiu_advice]
                suqiu_tree[ps] = tree
                # KIWI
                logging.info('前提不满足' + ' ' + str(suqiu))
                suqiu_debug_info = "诉求处理情况: 前提不满足。"
                debug_info[str(suqiu)] = suqiu_debug_info
            else:
                # KIWI
                logging.info('前提不满足且不在logic_ps_result中。' + ' ' + str(suqiu))
                suqiu_debug_info = "诉求处理情况: 前提不满足且不在logic_ps_result中。"
                debug_info[str(suqiu)] = suqiu_debug_info
        elif next[0] == 'factor':  # 有factor不确定的情况
            question_next = factor_question_dict[ps][next[1]]
            question_type = '1' if question_next not in question_multiple_dict[ps] else '2'
            # KIWI
            logging.info('前置factor不确定' + ' factor:' + str(next[1]))
            suqiu_debug_info = '诉求处理情况: 前置特征不确定。'
            next_question_debug_info = "对前置特征【{}】提问，问题来自特征表。".format(str(next[1]))
            suqiu_debug_info = suqiu_debug_info + '\n' + next_question_debug_info
            debug_info[str(suqiu)] = suqiu_debug_info
            break
        else:  # next = ('suqiu', suqiu)
            tree = LogicTree(problem, next[1], debug)

            # 将特征结果加入
            for factor, flag in suqiu_factor.items():  # 特征默认值
                tree.add_match_result(factor, flag, None)
            for factor, sentence in factor_sentence_list.items():  # 用户输入的特征
                sentence_matched, flag = sentence
                tree.add_match_result(factor, flag, sentence_matched)
            for question, answers in question_answers.items():  # 问答的输入特征
                factors = tree.add_question_result(question, answers.split(';'))
                question_answer_str += '。'.join(factors) + '。'

            next_question_debug_info = []
            question_next = tree.get_next_question(next_question_debug_info)
            if question_next is not None:
                # 如果是候选问题，应该根据候选问答表，来决定question_type的值。
                if tree.next_question_is_candidate_question():
                    question_type = '1' if question_next not in candidate_multiple_dict[ps] else '2'
                # 否则，应该根据特征表，来决定question_type的值。
                else:
                    question_type = '1' if question_next not in question_multiple_dict[ps] else '2'

                # KIWI
                logging.info('对诉求提问' + ' suqiu:' + str(next[1]))
                suqiu_debug_info = '诉求处理情况: 对诉求提问。'
                suqiu_debug_info = suqiu_debug_info + '\n' + '\n'.join(next_question_debug_info)
                debug_info[str(suqiu)] = suqiu_debug_info
                break
            else:
                result, _, _, _, _ = tree.get_logic_result()
                suqiu_result[suqiu] = result
                suqiu_tree[ps] = tree
                for factor, flag in tree.factor_flag.items():
                    if flag == 0:
                        continue
                    if factor in factor_sentence_list:
                        continue
                    factor_sentence_list[factor] = [tree.factor_sentence[factor], flag]

                # KIWI
                logging.info('诉求有结论' + ' suqiu: ' + str(next[1]) + '\t' + 'result: ' + str(result))
                suqiu_debug_info = '诉求处理情况: 诉求有结论。' + '诉求: ' + str(next[1]) + '\t' + 'result: ' + str(result)
                suqiu_debug_info = suqiu_debug_info + '\n' + '\n'.join(next_question_debug_info)
                debug_info[str(suqiu)] = suqiu_debug_info

    # 5. 如果没有问题了，那么需要出评估报告
    logging.info('5.5. return result dict')
    result_dict = {}
    report_dict = {}
    if question_next is None:
        for prob_problem_suqiu in prob_problem_suqius:
            has_result = False
            for logic_problem_suqiu in prob_ps2logic_ps[prob_problem_suqiu]:
                if logic_problem_suqiu in suqiu_tree:
                    has_result = True
            if not has_result:
                continue

            problem, prob_suqiu = prob_problem_suqiu.split('_')
            inputs = prob_ps_desc[prob_problem_suqiu] + '。' + fact + '。' + question_answer_str
            # inputs = fact + '。' + question_answer_str
            if prob_problem_suqiu in ['交通事故_违章扣分', '劳动社保_确认劳动劳务关系', '知识产权_确认著作权归属']:
                possibility_support = None
            else:
                # possibility_support = None
                possibility_support = predict(problem, prob_suqiu, inputs)
            logging.info('%s support probability: %s' % (prob_suqiu, possibility_support))

            result_value = []
            reason_of_evaluation = []
            evidence_module = []
            legal_advice = []
            support_value = []
            for logic_problem_suqiu in prob_ps2logic_ps[prob_problem_suqiu]:
                tree = suqiu_tree[logic_problem_suqiu]
                result, reason, proof, advice, support = tree.get_logic_result()
                if logic_problem_suqiu not in logic_ps_condition or logic_ps_condition[logic_problem_suqiu] == result:
                    result_value.append(result)
                    reason_of_evaluation.append(reason)
                    evidence_module.append(proof)
                    legal_advice.append(advice)
                    support_value.append(support)

            if len(result_value) == 0:
                continue

            result = result_value[0]
            reason = '\n\n'.join(reason_of_evaluation)
            proof = evidence_module[0]
            advice = legal_advice[0]
            support = support_value[0]

            support_or_not = '支持' if result == 1 else '不支持'
            reason, possibility_support = reason_probability_correct(prob_suqiu, inputs, support, reason,
                                                                     possibility_support, debug)
            report_dict[prob_suqiu] = {'reason_of_evaluation': reason,
                                       'evidence_module': proof,
                                       'legal_advice': advice,
                                       'possibility_support': as_num(possibility_support),
                                       'support_or_not': support_or_not}

    result_dict['question_asked'] = question_answers  # 问过的问题
    result_dict['question_next'] = question_next  # 下一个要问的问题
    result_dict['question_type'] = question_type  # 下一个要问的问题的类型

    result_dict['factor_sentence_list'] = [[s[0], f, s[1], ''] for f, s in factor_sentence_list.items()]  # 匹配到短语的列表，去重
    result_dict['result'] = report_dict  # 评估报告，包括评估理由、证据模块、法律建议、支持与否

    result_dict['debug_info'] = debug_info  # 记录中间信息，方便定位问题
    return result_dict


def as_num(x):  # format condidence_score
    y = '{:.3f}'.format(x)  # 5f表示保留5位小数点的float型
    return float(y)


def reason_probability_correct(suqiu, inputs, support, reason, probability, debug):
    np.random.seed(len(re.findall('[\u4E00-\u9FA5]', inputs)) + len(suqiu))
    if probability is None:
        if support == '支持':
            return reason, np.random.random() * 0.2 + 0.65
        else:
            return reason, np.random.random() * 0.2 + 0.25
    if support == '支持' and probability < 0.5:
        reason += '\n\n需要注意的是，基于裁判文书中相似案例的分析结果显示，您的诉求支持率较低，可能是因为存在【酌定情节】，或是很多案件在庭审过程中【举证不足】，导致法院驳回诉讼请求。'
        probability = 0.3 + probability / 2.5
    elif support == '不支持' and probability > 0.5:
        probability = 0.3 + (probability - 0.5) / 3
        # if reason in ['您的情形与法律支持情形匹配度较低，可能无法得到支持。', '缺少关键信息，无法给出准确评估。']:
        #     probability = probability - 0.15
        # else:
        #     reason += '\n\n值得注意的是，基于裁判文书中相似案例的分析结果显示，您的诉求支持率较高，可能是因为存在【酌定情节】，或法律支持的【其他情形】，建议您咨询律师或重新评估。'
    elif support == '不满足前提':
        probability = probability / 2

    probability = min(probability, 0.85 + np.random.random() * 0.05)  # 最后输出不会超过0.93
    probability = max(probability, 0.1 + np.random.random() * 0.05)
    if not debug:
        reason = text_underline(reason)
    return reason, probability


if __name__ == '__main__':
    # claim_list=["离婚", "财产分割"] # "离婚",
    # fact="男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。"
    # problem="婚姻家庭"
    # factor_sentence_list= []
    # question_answers={}
    # result_0 = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)
    # print("###第0次调用输入。problem:", problem, ";claim_list:", claim_list, ";fact:", fact, ";question_answers:", question_answers, ";factor_sentence_list:", factor_sentence_list)
    # print("###第0次调用结果:", result_0)

    # # # 第一次调用
    # problem = '婚姻家庭'
    # claim_list = ["离婚"]
    # fact="男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。"
    # question_answers = {}  # {'房子登记在谁的名下？:您;对方;双方;其他人':'对方','由谁付的？:您;对方;双方;您父母;双方父母;对方父母':'对方父母'}
    # factor_sentence_list = []
    # result_1 = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)
    # print("###第一次调用输入。problem:", problem, ";claim_list:", claim_list, ";fact:", fact, ";question_answers:", question_answers, ";factor_sentence_list:", factor_sentence_list)
    # print("###第一次调用结果:", result_1)
    # print("---------------------------------------------------------------------------")

    # # 第二次调用
    # problem = '婚姻家庭'
    # claim_list = ['房产分割']
    # fact = '婚后男的方父母出资首得到付，夫妻名义贷款还贷，房产证只写男方名，离婚后财产如何分配'
    # question_answers = {'由谁付的首付？:您;对方;双方;您父母;双方父母;对方父母': '对方父母'}
    # factor_sentence_list = [['婚后男的方父母出资首得到付', '婚后购买', 1, ''], ['房产证只写男方名', '有房产证', 1, ''], ['房产证只写男方名', '登记在对方名下', 1, ''], ['夫妻名义贷款还贷', '首付', 1, '']]
    # result_2 = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)
    # print("####第二次调用输入。problem:", problem, ";claim_list:", claim_list, ";fact:", fact, ";question_answers:", question_answers, ";factor_sentence_list:", factor_sentence_list)
    # print("###第二次调用结果:", result_2)
    # print("---------------------------------------------------------------------------")

    # 第三次调用
    # problem = '婚姻家庭'
    # claim_list = ['房产分割']
    # fact = '婚后男的方父母出资首得到付，夫妻名义贷款还贷，房产证只写男方名，离婚后财产如何分配'
    # question_answers = {'由谁付的首付？:您;对方;双方;您父母;双方父母;对方父母': '对方父母','房子登记在谁的名下？:男方;女方;双方;其他人':'男方'}
    # factor_sentence_list = [['婚后男的方父母出资首得到付', '婚后购买', 1, ''], ['房产证只写男方名', '有房产证', 1, ''], ['房产证只写男方名', '登记在对方名下', 1, ''], ['夫妻名义贷款还贷', '首付', 1, '']]
    # result_2 = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)
    # print("####第三次调用输入。problem:", problem, ";claim_list:", claim_list, ";fact:", fact, ";question_answers:", question_answers, ";factor_sentence_list:", factor_sentence_list)
    # print("###第三次调用结果:", result_2)
    # print("---------------------------------------------------------------------------")

    # claim_list=["减少租金或者不支付租金"] # "离婚",
    # fact="减少租金可以吗"
    # problem="租赁纠纷"
    # factor_sentence_list= []
    # question_answers={}
    # result_0 = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)
    # print("###第0次调用输入。problem:", problem, ";claim_list:", claim_list, ";fact:", fact, ";question_answers:", question_answers, ";factor_sentence_list:", factor_sentence_list)
    # print("###第0次调用结果:", result_0)

    claim_list = ["减少租金或者不支付租金"]  # "离婚",
    fact = "11111111"
    problem = "租赁合同"
    factor_sentence_list = []
    question_answers = {}
    result_0 = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)
    print("###第0次调用输入。problem:", problem, ";claim_list:", claim_list, ";fact:", fact, ";question_answers:",
          question_answers, ";factor_sentence_list:", factor_sentence_list)
    print("###第0次调用结果:", result_0)
