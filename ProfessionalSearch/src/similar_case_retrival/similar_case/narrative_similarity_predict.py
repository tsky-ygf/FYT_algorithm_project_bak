import random
import re

from elasticsearch import Elasticsearch
import jieba
import logging
import jieba.analyse
import time

from ProfessionalSearch.src.similar_case_retrival.similar_case.text_cnn_predict import get_feature
from ProfessionalSearch.src.similar_case_retrival.similar_case.question_answering_ask_type_predict import (
    get_question_feature_label_prob,
    get_appeal_by_rules,
)
from ProfessionalSearch.src.similar_case_retrival.similar_case.identify_problem_suqiu import (
    predict_problem_suqiu,
)



from ProfessionalSearch.src.similar_case_retrival.similar_case.rank_util import (
    pseg_txt,
    compute_sentence_embedding,
    cosine_similiarity,
    as_num, load_vocab_embedding_idf,
)

print("abc2...")

ip = "192.168.1.254"
es = Elasticsearch()  # [ip],http_auth=('elastic', 'password'),port=9200
# ES_IDX_NAME = "narrative_similarity_v2"  # 'narrative_similarity_v2'
ES_IDX_NAME = "case_index_minshi_v2"  # 'narrative_similarity_v2'
print("end...")
# Predefine pos scope
NOT_FOUND = "没有相似案件"
similiar_threshold = 0.50  # 0.2 #80
similiar_threshold_consult = 0.65
NUM_MAX_CANDIDATES = 30  # 50
NUM_MAX_CANDIDATES_CONSULT = 10

data_path = "../data/question_answering/"  # TODO 暂时先用一下看看效果
vocab_file = data_path + "consult_vocab.csv"  # TODO 词汇表
idf_file = data_path + "consult_idf_dict2.csv"  # TODO idf的缓存文件，暂时用不到。
vocab_embedding_file = data_path + "model_law_query.vec"  # TODO  使用fastext训练的词向量
# vocab_list, vocab_dict,vcoab_embedding, idfs_dict,tfidf_vectorizer,tfidf_vocab_dict=load_vocab_embedding_idf(data_path,vocab_file,vocab_embedding_file,idf_file)
# 加载bert模型

problem_suqiu_conversion = {
    "工伤赔偿_认定工伤": "工伤赔偿_工伤赔偿",
    "工伤赔偿_支付工伤期间工资": "工伤赔偿_工伤赔偿",
    "工伤赔偿_支付工伤保险待遇": "工伤赔偿_工伤赔偿",
    "婚姻家庭_解除收养关系": "婚姻家庭_",
    "婚姻家庭_财产分割": "婚姻家庭_财产分割",
    "婚姻家庭_抚养费": "婚姻家庭_支付抚养费",
    "婚姻家庭_扶养费": "婚姻家庭_",
    "婚姻家庭_赡养费": "婚姻家庭_支付赡养费",
    "婚姻家庭_返还彩礼": "婚姻家庭_返还彩礼",
    "婚姻家庭_婚姻无效": "婚姻家庭_确认婚姻无效",
    "婚姻家庭_监护权": "婚姻家庭_",
    "婚姻家庭_探望权": "婚姻家庭_行使探望权",
    "婚姻家庭_离婚": "婚姻家庭_离婚",
    "婚姻家庭_抚养权": "婚姻家庭_确认抚养权",
    "继承问题_确认遗嘱有效": "继承问题_遗产继承",
    "继承问题_确认继承权": "继承问题_遗产继承",
    "继承问题_遗产分配": "继承问题_遗产继承",
    "继承问题_偿还债务": "继承问题_遗产继承",
    "继承问题_确认遗嘱无效": "继承问题_遗产继承",
    "交通事故_赔偿损失": "交通事故_损害赔偿",
    "借贷纠纷_借贷有效": "借贷纠纷_还本付息",
    "借贷纠纷_偿还本金": "借贷纠纷_还本付息",
    "借贷纠纷_解除合同": "借贷纠纷_还本付息",
    "借贷纠纷_支付利息": "借贷纠纷_还本付息",
    "借贷纠纷_借贷无效": "借贷纠纷_还本付息",
    "金融借款合同纠纷_支付利息": "借贷纠纷_还本付息",
    "金融借款合同纠纷_归还借款": "借贷纠纷_还本付息",
    "劳动纠纷_支付工资": "劳动劳务_支付劳动劳务报酬",
    "劳动纠纷_支付经济补偿金": "劳动劳务_经济补偿金或赔偿金",
    "劳动纠纷_支付赔偿金": "劳动劳务_经济补偿金或赔偿金",
    "劳动纠纷_支付加班工资": "劳动劳务_支付加班工资",
    "劳动纠纷_确认存在劳动关系": "劳动劳务_",
    "劳动纠纷_支付二倍工资": "劳动劳务_支付双倍工资",
    "劳动纠纷_支付竞业限制违约金": "劳动劳务_",
    "劳务纠纷_支付劳务报酬": "劳动劳务_支付劳动劳务报酬",
    "社保纠纷_返还垫付的养老保险费": "社保纠纷_社保待遇",
    "社保纠纷_赔偿养老保险待遇损失": "社保纠纷_社保待遇",
    "社保纠纷_赔偿医疗保险待遇损失": "社保纠纷_社保待遇",
    "社保纠纷_返还垫付的医疗保险费": "社保纠纷_社保待遇",
    "社保纠纷_生育保险待遇": "社保纠纷_社保待遇",
    "社保纠纷_失业保险待遇": "社保纠纷_社保待遇",
    "提供劳务者受害责任纠纷_赔偿损失": "劳动劳务_劳务受损赔偿",
    "提供劳务者致害责任纠纷_赔偿损失": "劳动劳务_劳动劳务致损赔偿",
    "义务帮工人受害责任纠纷_赔偿损失": "劳动劳务_劳务受损赔偿",
    "用人单位责任纠纷_赔偿损失": "劳动劳务_劳动劳务致损赔偿",
}

label_convert = {
    "公司法": "公司业务",
    "合同纠纷": "合同纠纷",
    "刑事辩护": "刑事辩护",
    "债权债务": "银行借贷",
    "房产纠纷": "房产物业",
    "劳动纠纷": "劳动劳务",
    "婚姻家庭": "婚姻家庭",
    "交通事故": "交通事故",
    "other": "other",
    "建设工程": "建设工程",
    "侵权纠纷": "侵权纠纷",
    "知识产权": "知识产权",
    "医疗纠纷": "医疗纠纷",
}

new_label_list = list(label_convert.keys())
logging.info("new_label_list:{}".format(new_label_list))
print("new_label_list:{}".format(new_label_list))


def case_old2new(case, appeal):
    new_case = case
    new_appeal = []
    for i in appeal:
        temp_str = case + "_" + i
        flag = True
        for key, value in problem_suqiu_conversion.items():
            if temp_str == key:
                temp = value.split("_")
                new_case = temp[0]
                flag = False
                if temp[1] == "":
                    continue
                new_appeal.append(temp[1])
                print(new_appeal)
                break
        if flag:
            new_appeal.append(i)
    return new_case, list(set(new_appeal))


def predict_fn(fact, problem="", claim_list=[]):
    """
    输入一个问题，返回一个答案，和相似问题的ID列表
    :param question:
    :return:
    """
    is_consult_flag = False  # is_consult_flag：是否是咨询。true:是咨询；默认是评估
    if problem == "" or len(claim_list) == 0:
        is_consult_flag = True  # 是咨询
    print(
        "predict_fn.fact:"
        + fact
        + ";problem:"
        + problem
        + ";claim_list:"
        + str(claim_list)
    )
    logging.info(
        "predict_fn.fact:"
        + fact
        + ";problem:"
        + problem
        + ";claim_list:"
        + str(claim_list)
    )
    # 1.预测问题类型、诉求、问题类型的置信区间
    if problem == "" and claim_list == []:  # 如果问题类型和诉求都为空的话，那么先做意图识别
        problem_suqiu_type_dict = predict_problem_suqiu(fact)
        problem = problem_suqiu_type_dict.get("case_reason", "")
        case_reason_confidence = str(
            problem_suqiu_type_dict.get("case_reason_confidence", "")
        )
        claim_list = problem_suqiu_type_dict.get("appeal", [])
        logging.info(
            "case_reason:{}, appeal:{}, confidence:{},".format(
                problem, str(claim_list), case_reason_confidence
            )
        )
        _, label, p = get_question_feature_label_prob(fact)
        label_1 = label_convert.get(label, label)
        label, my_appeal = get_appeal_by_rules(fact, label_1)

        logging.info("label:{}".format(label))
        print("label:{}".format(label))
        if (label != "other" and p > 0.85) or (label == "other" and p > 0.95):
            if problem != label:
                claim_list = my_appeal
            problem = label
            case_reason_confidence = p

        problem, claim_list = case_old2new(problem, claim_list)
        logging.info(
            "problem:{}, appeal:{}, confidence:{},".format(
                problem, str(claim_list), case_reason_confidence
            )
        )

    # 2. 召回
    t1 = time.time()
    candidate_list, tags = search_similar_case(
        fact, is_consult_flag, problem=problem, claim_list=claim_list
    )

    # each element of candidate_list: (doc_id, problem_type, suqiu_type, suqiu_label, suqing_sentences, chaming,benyuan_renwei,tags)

    # 3. 对候选列表进行排序
    similiar_list_ = candidate_list_q2q_rank(fact, candidate_list)
    # similiar_list_=rank_candiate_list_v01(fact, candidate_list) # question-question similiarity
    t2 = time.time()
    print(
        "predict_fn.1.time spent for recall candidate list and compute similairity of questions:",
        (t2 - t1),
    )

    # 4. 组织返回结果：1）根据阀值情况设置答案；2）暂时移除重复的问题
    result = organize_result(similiar_list_, is_consult_flag, fact, tags)
    return result


def candidate_list_q2q_rank(original_question, candidate_list):
    """
    相似类案：问题和问题的排序
    :param candidate_list:
    :return: a new candidate_list. each element should contain: (doc_id, cos_i, win_los, problem_type, suqiu_type, tags)
    """
    original_question = pseg_txt(original_question).replace(" ", "")  # 过滤一部分数据
    # original_question=original_question[0:77]
    logging.info(
        "narrative_similarity_predict.candidate_list_q2q_rank."
        + "original_question.filtered:"
        + original_question
    )
    # feature_q,_,_=predict_online_q2type(original_question) todo replace temp 2019.05.15 bright.xu
    feature_q = get_feature(original_question)

    candidate_list_new = []
    for i, element in enumerate(candidate_list):
        (
            uq_id,
            problem_type,
            suqiu_type,
            # suqiu_label,
            sucheng_sentences,
            chaming,
            benyuan_renwei,
            tags,
            event_num,
        ) = element
        eventDate = re.findall("(?<=\（)(.+?)(?=\）)", event_num)
        candidate = pseg_txt(chaming).replace(" ", "")  # 过滤一部分数据
        print(
            "narrative_similarity_predict.candidate_list_q2q_rank.",
            i,
            "candidate:",
            candidate,
        )
        logging.info(
            "narrative_similarity_predict.candidate_list_q2q_rank."
            + str(i)
            + "candidate.filtered:"
            + candidate
        )
        # feature_c, label_predict_c, possibility_c=predict_online_q2type(candidate) todo replace temp 2019.05.15 bright.xu
        feature_c = get_feature(candidate)

        cos_i = cosine_similiarity(feature_q, feature_c)
        # win_los = 1 if "1" in suqiu_label else 0  # TODO 有一个诉求得到支持，暂时任务就是支持的。
        logging.info("cos_i:"+ str(cos_i))
        candidate_list_new.append(
            (uq_id, cos_i, problem_type, suqiu_type, tags, eventDate)
        )

    candidate_list_new = sorted(
        candidate_list_new, key=lambda element: (element[5], -element[1]), reverse=True
    )  # 排序
    logging.info(
        "narrative_similarity_predict.candidate_list_q2q_rank.candidate_list_new:"
        + str(candidate_list_new)
    )
    print(
        "narrative_similarity_predict.candidate_list_q2q_rank.candidate_list_q2q_rank:"
        + str(candidate_list_new)
    )
    return candidate_list_new


def rank_candiate_list_v01(question_original, similiar_question_list):
    """
    v0.1版本：结合tfidf和词向量，计算原始问题和候选问题的相似性
    :param similiar_question_list:
    :return: 排序后的列表
    """
    # 1.原始问题计算句子向量
    # question_original=" ".join(jieba.lcut(question_original))
    question_pseg = pseg_txt(question_original)
    print("cases.rank_candiate_list_v01:", question_original)
    # TODO add tags
    emb_query, query_filtered = compute_sentence_embedding(
        question_pseg,
        vocab_dict,
        vcoab_embedding,
        idfs_dict,
        tfidf_vectorizer,
        tfidf_vocab_dict,
    )

    # 2.计算候选项的句子向量
    similiar_list = []
    for element in similiar_question_list:
        # 使用tags和(过滤过的)查明去计算相似性
        (
            doc_id,
            problem_type,
            suqiu_type,
            suqiu_label,
            suqing_sentences,
            chaming,
            benyuan_renwei,
            tags,
        ) = element
        candidate_sentences = tags + " " + chaming
        emb_question, _ = compute_sentence_embedding(
            candidate_sentences,
            vocab_dict,
            vcoab_embedding,
            idfs_dict,
            tfidf_vectorizer,
            tfidf_vocab_dict,
        )
        # 候选问题计算句子向量
        cos_i = cosine_similiarity(emb_query, emb_question)
        win_los = 1 if "1" in suqiu_label else 0  # TODO 有一个诉求得到支持，暂时任务就是支持的。
        similiar_list.append(
            (doc_id, as_num(cos_i), win_los, problem_type, suqiu_type, tags)
        )

    # 3.根据相似性结果排序
    similiar_list = sorted(similiar_list, key=lambda element: element[1], reverse=True)
    logging.info("cases.rank_candiate_list_v01.similiar_list:" + str(similiar_list))
    print("cases.rank_candiate_list_v01.similiar_list:", similiar_list)

    return similiar_list


problem_suqiu_conversion2 = {
    "婚姻家庭_离婚": ["婚姻家庭_离婚"],
    "婚姻家庭_确认婚姻无效": ["婚姻家庭_婚姻无效"],
    "婚姻家庭_返还彩礼": ["婚姻家庭_返还彩礼"],
    "婚姻家庭_财产分割": ["婚姻家庭_财产分割"],
    "婚姻家庭_房产分割": ["婚姻家庭_财产分割"],
    "婚姻家庭_夫妻共同债务": ["婚姻家庭_财产分割"],
    "婚姻家庭_确认抚养权": ["婚姻家庭_抚养权"],
    "婚姻家庭_支付抚养费": ["婚姻家庭_抚养费"],
    "婚姻家庭_增加抚养费": ["婚姻家庭_抚养费"],
    "婚姻家庭_减少抚养费": ["婚姻家庭_抚养费"],
    "婚姻家庭_行使探望权": ["婚姻家庭_探望权"],
    "婚姻家庭_支付赡养费": ["婚姻家庭_赡养费"],
    "继承问题_遗产继承": ["继承问题_确认继承权", "继承问题_遗产分配", "继承问题_确认遗嘱有效", "继承问题_偿还债务"],
    "社保纠纷_社保待遇": ["社保纠纷_养老保险赔偿", "社保纠纷_失业保险赔偿"],
    "工伤赔偿_工伤赔偿": ["工伤赔偿_因工受伤赔偿", "工伤赔偿_因工致残赔偿"],
    "劳动劳务_劳动劳务致损赔偿": ["提供劳务者致害责任纠纷_赔偿损失"],
    "劳动劳务_劳务受损赔偿": ["提供劳务者受害责任纠纷_赔偿损失", "义务帮工人受害责任纠纷_赔偿损失"],
    "劳动劳务_支付劳动劳务报酬": ["劳动纠纷_支付工资", "劳务纠纷_支付劳务报酬", "劳动纠纷_确认存在劳动关系"],
    "劳动劳务_支付加班工资": ["劳动纠纷_支付加班工资"],
    "劳动劳务_经济补偿金或赔偿金": ["劳动纠纷_支付经济补偿金", "劳动纠纷_支付赔偿金"],
    "劳动劳务_支付双倍工资": ["劳动纠纷_支付二倍工资"],
    "交通事故_损害赔偿": ["交通事故_赔偿损失"],
    "借贷纠纷_还本付息": ["借贷纠纷_偿还本金", "借贷纠纷_支付利息", "金融借款合同纠纷_归还借款", "金融借款合同纠纷_支付利息"],
}


def search_similar_case(
    question_original, is_consult_flag, problem="", claim_list=[], top_k=15
):
    """
    召回模型：通过问题召回候选问题和答案列表,返回最多前10个元素的列表
    :param question_original:
    :return:
    """
    t1 = time.time()
    logging.info("search_similar_case.question_original:" + question_original)
    if len(question_original) > 400:  # 截取超长的查询
        question_original = question_original[0:200] + "。" + question_original[-200:]
    question = " ".join(jieba.lcut(question_original))
    question_short = pseg_txt(question_original)
    print("search_similar_case.question_short:", question_short, ";quesiton:", question)

    # 纠纷类型和诉求类型权重高一点；然后是chaming、benyuan_renwei、
    tags1 = jieba.analyse.extract_tags(question_original, topK=top_k)
    tags2 = jieba.analyse.textrank(
        question_original, topK=top_k, withWeight=False, allowPOS=("ns", "n", "vn", "v")
    )
    tags = list(set(tags1).intersection(set(tags2)))
    tags = " ".join(tags)
    logging.info(
        "search_similar_case.question_short:"
        + question_short
        + ";tags:"
        + tags
        + ";quesiton:"
        + question
    )

    question_last = question[-30:]
    query_search = (
        " ".join(tags) + " " + question_last + " " + question_short
    )  # query_search包括关键词、后面的比较重要的部分、过滤后的数据
    print("")
    # claim_list="x1,x2"
    if isinstance(claim_list, str):
        claim_list = claim_list.split(",")
    if isinstance(claim_list, list):
        claim_list.sort()

    suqiu_list = []
    for suqiu in claim_list:
        if problem + "_" + suqiu in problem_suqiu_conversion2:
            suqiu_list += problem_suqiu_conversion2[problem + "_" + suqiu]
        else:
            suqiu_list += [problem + "_" + suqiu]
    if len(suqiu_list) > 0:
        problem = random.choice(suqiu_list).split("_")[0]
        claim_list = [
            ps.split("_")[1] for ps in suqiu_list if ps.startswith(problem + "_")
        ]
    elif problem == "劳动劳务":
        problem = "劳动纠纷"
    suqiu_type = (
        "" if len(claim_list) <= 0 else " ".join(claim_list)
    )  # 如果诉求类型列表为空，则搜索空，否则排序后搜索

    logging.info(
        "search_similar_case.problem:"
        + problem
        + ";suqiu_type:"
        + suqiu_type
        + "; original.claim_list:"
        + str(claim_list)
    )
    print(
        "search_similar_case.problem:"
        + problem
        + ";suqiu_type:"
        + suqiu_type
        + "; original.claim_list:"
        + str(claim_list)
    )

    body = get_search_body(problem, suqiu_type, query_search, tags)
    search_size = NUM_MAX_CANDIDATES_CONSULT if is_consult_flag else NUM_MAX_CANDIDATES
    res = es.search(
        index=ES_IDX_NAME, size=search_size, body=body
    )  # res = es.search(index="test-index", body={"query": {"match_all": {}}})
    logging.info("search_similar_case.total hits:" + str(res["hits"]["total"]))
    count = 0
    candidate_list = []
    for hit in res["hits"]["hits"]:
        _source = hit["_source"]
        sucheng_sentences = _source["sucheng_sentences"]
        chaming = _source["chaming"]
        benyuan_renwei = _source["benyuan_renwei"]
        problem_type = _source["jfType"]
        # suqiu_type = _source["suqiu_type"]
        # suqiu_label = _source["suqiu_label"]
        doc_id = _source["uq_id"]
        event_num = _source["event_num"]
        tags_ = hit["_source"]["tags"]
        if not sucheng_sentences:
            sucheng_sentences = ""
        if not chaming:
            chaming = ""
        if not benyuan_renwei:
            benyuan_renwei = ""
        if not problem_type:
            problem_type = ""
        if count < NUM_MAX_CANDIDATES:
            logging.info(
                count,
                "--->",
                doc_id,
                ";problem_type:",
                problem_type,
                # ";suqiu_type:",
                # suqiu_type,
                # ";suqiu_label:",
                # suqiu_label,
                ";tags_:",
                tags_,
                ";event_num:",
                event_num,
            )
            logging.info(
                str(doc_id)
                + ";problem_type:"
                + problem_type
                # + ";suqiu_type:"
                # + suqiu_type
                # + ";suqiu_label:"
                # + suqiu_label
                + ";tags_:"
                + tags_
                +";event_num:"
                +event_num
            )
            print(
                count,
                "suqing_sentences:",
                sucheng_sentences,
                ";chaming:",
                chaming,
                ";benyuan_renwei:",
                benyuan_renwei,
            )
            logging.info(
                "suqing_sentences:"
                + sucheng_sentences
                + ";chaming:"
                + chaming
                + ";benyuan_renwei:"
                + benyuan_renwei
            )
            # TODO 在搜索结果中移除与纠纷类型、诉求无关的数据
        #     if problem != "" and len(problem) > 0:
        #         if problem_type == problem:
        #             if len(claim_list) == 0 or suqiu_type in claim_list:
        #                 candidate_list.append(
        #                     (
        #                         doc_id,
        #                         problem_type,
        #                         suqiu_type,
        #                         # suqiu_label,
        #                         yg_sc_sentences,
        #                         chaming,
        #                         benyuan_renwei,
        #                         # tags_,
        #                     )
        #                 )  # did_temp,score_temp,question_temp,answer_temp,question_short_temp
        #         else:
        #             print(
        #                 "search_similar_case.removing.data from search enginee:"
        #                 + str(
        #                     (
        #                         doc_id,
        #                         problem_type,
        #                         suqiu_type,
        #                         # suqiu_label,
        #                         yg_sc_sentences,
        #                         chaming,
        #                         benyuan_renwei,
        #                         # tags_,
        #                     )
        #                 )
        #             )
        #             logging.info(
        #                 "search_similar_case.removing.data from search enginee:"
        #                 + str(
        #                     (
        #                         doc_id,
        #                         problem_type,
        #                         suqiu_type,
        #                         # suqiu_label,
        #                         yg_sc_sentences,
        #                         chaming,
        #                         benyuan_renwei,
        #                         # tags_,
        #                     )
        #                 )
        #             )
        #     else:  # problem and claim_list is empty
        candidate_list.append(
            (
                doc_id,
                problem_type,
                suqiu_type,
                # suqiu_label,
                sucheng_sentences,
                chaming,
                benyuan_renwei,
                tags_,
                event_num,
            )
        )  # did_temp,score_temp,question_temp,answer_temp,question_short_temp
        #
        count = count + 1

    t2 = time.time()
    print("search_similar_case.length of candidate_list:", len(candidate_list))
    logging.info(
        "search_similar_case.length of candidate_list:"
        + str(len(candidate_list))
        + ";time spent:"
        + str(t2 - t1)
    )
    return candidate_list, tags


def get_search_body(problem, suqiu_type, query_search, tags):
    """
    search body
    :param problem:
    :param suqiu_type:
    :param query_search:
    :param tags:
    :return:
    """
    if (
        problem is not None
        and problem != ""
        # and suqiu_type is not None
        # and suqiu_type != ""
    ):
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"jfType": {"query": problem, "boost": 10}}},
                        # {"match": {"suqiu_type": {"query": suqiu_type, "boost": 10}}},
                        {
                            "bool": {
                                "should": [
                                    {"match": {"chaming": query_search}},
                                    {"match": {"benyuan_renwei": query_search}},
                                    {"match": {"sucheng_sentences": query_search}},
                                    # {"match": {"bg_sc": query_search}},
                                    # {"match": {"tags": tags}},
                                ]
                            }
                        },
                    ]
                }
            }
        }
    else:
        body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"chaming": query_search}},
                        {"match": {"benyuan_renwei": query_search}},
                        {"match": {"sucheng_sentences": query_search}},
                        # {"match": {"bg_sc": query_search}},
                        # {"match": {"tags": tags}},
                    ]
                }
            }
        }
    logging.info(body)
    return body


def organize_result(similiar_list, is_consult_flag, fact, keywords, top_k=50):
    """
    组织答案，包括：1.根据阀值情况设置答案；2.暂时移除重复的问题
    :param similiar_list:
    :param top_k:
    :return:
    """
    (
        doc_id_list,
        sim_list,
        win_los_list,
        reason_name_list,
        appeal_name_list,
        tags_list,
        date_list
    ) = ([], [], [], [], [], [], [])
    for i, element in enumerate(similiar_list):
        doc_id, cos_i, problem_type, suqiu_type, tags, pubDate = element
        tags = tags.replace("本院", "判定")
        threshold = (
            similiar_threshold_consult if is_consult_flag else similiar_threshold
        )  # similiar_threshold_consult
        if float(cos_i) >= float(threshold):
            doc_id_list.append(str(doc_id))
            sim_list.append(float(cos_i))
            # win_los_list.append(win_los)
            reason_name_list.append(str(problem_type))
            appeal_name_list.append(str(suqiu_type))
            tags_list.append(tags)
            date_list.append(pubDate)

    tags1 = jieba.analyse.extract_tags(fact, topK=20)
    tags2 = jieba.analyse.textrank(
        fact, topK=20, withWeight=False, allowPOS=("nz", "nt", "n", "vn", "v")
    )

    return (
        doc_id_list,
        sim_list,
        # win_los_list,
        reason_name_list,
        appeal_name_list,
        tags_list,
        keywords,
        date_list,
    )


def remove_duplicate_candiate_list(similiar_question_list):
    """
    v0版本的候选排序：只做了相同问题的去重。如果有两个相同的问题，那么质量分低的将被过滤掉。
    :param similiar_question_list:
    :return:
    """
    similiar_question_list_new = []
    unique_question_dict = {}
    for element in similiar_question_list:
        qid, score, question, answer = element
        question_value = unique_question_dict.get(question, None)
        if question_value is not None:
            continue
        similiar_question_list_new.append((qid, question, score))
        unique_question_dict[question] = 1

    return similiar_question_list_new


if __name__ == "__main__":
    t1 = time.time()
    question_original = "社保待遇"
    result_dict = predict_fn(question_original, "社保纠纷", ["社保待遇"])
    print("result_dict:", result_dict)
    t2 = time.time()
    print(t2 - t1)
