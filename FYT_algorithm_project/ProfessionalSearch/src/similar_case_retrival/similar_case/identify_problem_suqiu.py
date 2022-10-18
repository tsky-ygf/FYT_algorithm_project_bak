import fasttext
import jieba
import re
import time
from ProfessionalSearch.src.similar_case_retrival.similar_case.rank_util import as_num

# 加载模型
base_path_fasttext = (
    "model/bxh_search_model/question_answering/bxh_identity_problem_and_suqiu.bin"
)
model = fasttext.load_model(base_path_fasttext, label_prefix="__label__")

case_reason_confidence_threshold = 0.65


def predict_problem_suqiu(query, threshold=0.23):  # 0.30
    """
    输入一个问题，预测问题类型和诉求类型
    :param problem: 用户输入的问题
    :return: 问题类型、诉求列表
    """
    # todo print("query:",query)
    query = re.sub("[^\u4E00-\u9FA5]", "", query)  # 过滤非中文字符
    if len(query) <= 5:
        return {}
    predict_result_list = model.predict_proba([" ".join(jieba.lcut(query))], k=3)[
        0
    ]  # [0][0] #[0][0], k=3)
    result_dict = process_result(predict_result_list, threshold)
    # todo print("result_dict:",result_dict)
    return result_dict


def process_result(predict_result_list, threshold):
    """
    处理预测的结果。 TODO 为简化期间，暂时先只考虑第一个
    :param predict_result_list: 预测的列表. e.g.  [('交通事故#赔偿损失', 0.366902), ('劳务纠纷#支付劳务报酬', 0.120723), ('工伤赔偿#因工受伤赔偿', 0.0748066)]
    :return: 问题类型、诉求类型列表、置信概率
    """
    result_dict = {}
    # todo print("predict_result_list:",predict_result_list)
    # todo logging.info("#预测的问题类型、诉求类型:"+str(predict_result_list))
    # first_element=predict_result_list[0] # ('交通事故#赔偿损失', 0.366902)
    # problem,suqiu=first_element[0].split("#")
    # possibility=first_element[1]
    # if possibility>=threshold:
    #    result_dict['case_reason']=problem
    #    result_dict['appeal']=suqiu
    #    result_dict['case_reason_confidence']=possibility
    case_reason = ""
    case_reason_confidence = 0.0
    appeal = []
    for i, element in enumerate(predict_result_list):
        problem_temp, suqiu_temp = element[0].split("#")
        possibility_temp = element[1]
        if possibility_temp > threshold:
            if i == 0:
                case_reason = problem_temp
            if problem_temp == case_reason:  # 设置问题类型的累加概率
                case_reason_confidence += possibility_temp
                appeal.append(suqiu_temp)

    # 设置返回值
    if case_reason_confidence >= case_reason_confidence_threshold:
        result_dict["case_reason"] = case_reason  # 纠纷类型
        result_dict["case_reason_confidence"] = as_num(case_reason_confidence)
        result_dict["appeal"] = appeal  # 诉求
    else:
        result_dict["case_reason"] = ""  # 纠纷类型
        result_dict["case_reason_confidence"] = as_num(0.0)
        result_dict["appeal"] = []  # 诉求
    return result_dict


if __name__ == "__main__":
    t1 = time.time()
    query = "打算离婚，想要孩子抚养权，但孩子都是妈妈带的，我要怎么办呢？"  #'今天我在公路上开车行驶时，到转弯的地方，被后面一辆摩托车追尾了，结果那个开摩托车的被撞伤了，我车子后面的玻璃都被撞碎了，那个摩托车司机被送医院了，我给交了几千块钱医疗费，等今天去交警大队处理时 ，那个摩托车司机还跟我要几万块的赔偿金，请问我该怎么办？？？请问我该付责任吗？？？'
    result_dict = predict_problem_suqiu(query)
    print("result_dict:", result_dict)
    t2 = time.time()
    result_dict = predict_problem_suqiu(query)
    t3 = time.time()
    print(t3 - t2)
