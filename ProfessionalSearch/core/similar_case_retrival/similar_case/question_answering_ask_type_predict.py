from ProfessionalSearch.core.similar_case_retrival.bert.tokenization import *
import os, re
import tensorflow as tf
import numpy as np
import pandas as pd

suqiu_keywords = pd.read_csv(
    "data/bxh_search_data/question_answering/common_config/suqiu_keywords.csv",
    index_col=None,
)
print("suqiu_keywords length:{}".format(len(suqiu_keywords)))

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def process_one_example(tokenizer, text_a, text_b=None, max_seq_len=256):
    """
        处理 单个样本
    """
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[0 : (max_seq_len - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    feature = (input_ids, input_mask, segment_ids)
    return feature


def load_model(model_folder):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except Exception as e:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder, repr(e))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    tf.reset_default_graph()
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(
        input_checkpoint + ".meta", clear_devices=clear_devices
    )

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    graph = tf.Graph().as_default()

    # We start a session and restore the graph weights
    sess = tf.Session()
    saver.restore(sess, input_checkpoint)
    return sess


tokenizer = FullTokenizer(
    "model/bxh_search_model/question_answering/bert_model_ask_type/vocab.txt"
)
sess = load_model("model/bxh_search_model/question_answering/bert_model_ask_type/")

input_ids = sess.graph.get_tensor_by_name("input_ids:0")
input_mask = sess.graph.get_tensor_by_name("input_mask:0")  # is_training
segment_ids = sess.graph.get_tensor_by_name(
    "segment_ids:0"
)  # fc/dense/Relu  cnn_block/Reshape
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")

f = sess.graph.get_tensor_by_name("bert/pooler/dense/Tanh:0")
p = sess.graph.get_tensor_by_name("loss/Softmax:0")  # BiasAdd

label2id = {
    "公司法": 0,
    "合同纠纷": 1,
    "刑事辩护": 2,
    "债权债务": 3,
    "房产纠纷": 4,
    "劳动纠纷": 5,
    "婚姻家庭": 6,
    "交通事故": 7,
    "other": 8,
    "建设工程": 9,
    "侵权纠纷": 10,
    "知识产权": 11,
    "医疗纠纷": 12,
}
id2label = {v: k for k, v in label2id.items()}

labor_key_words = (
    "工[伤资人地作]|[就入离辞降]职|评残|劳动合同|辞退|试用期|劳动仲裁|社保|[出考]勤|员工|[上下].{0,2}班|工作.*公司|"
    "劳动人事|[婚产丧病事年]假|加班费?|劳动能力|工作.*意外|劳动法|退休|降薪|解雇|解聘|失业保险|工作.*伤残|职业病|"
    "公司.*(解除|辞退?)|鉴定.*伤残|伤残.*鉴定|因工受伤|因工负伤|工作期间受伤|公司。*赔偿|裁员|开除|失业金|养老保险"
    "|旷工|伤残.*(赔偿|标准)|解除劳动|公积金|劳动纠纷|临时工|未评定伤残|用人单位|实习|年终奖|老板|劳动|雇佣|劳务|职工|"
    "补休|公伤鉴定|用工单位|法定假|调休|辞工书|辞工|被辞|工作期间"
)


def get_question_feature_label_prob(text):
    feature = process_one_example(tokenizer, text, None, 128)
    feed = {
        input_ids: [feature[0]],
        input_mask: [feature[1]],
        segment_ids: [feature[2]],
        keep_prob: 1.0,
    }
    feature, probs = sess.run([f, p], feed)
    max_index = int(np.argmax(probs[0]))
    prob = float(probs[0, max_index])
    label = id2label[max_index]
    if label == "劳动纠纷" and not re.search(labor_key_words, text):
        prob = 0.3
    return feature[0], id2label[max_index], prob


def get_appeal_by_rules(text, model_name):
    """
    使用规则匹配纠纷 对应的诉求
    """
    #     appeal = []
    #     if model_name == "婚姻家庭":
    #         if re.search("财产.*分[割配]",text):
    #             appeal.append("财产分割")
    #         if re.search("房产.*分[割配]",text):
    #             appeal.append("房产分割")
    #         if re.search("债",text):
    #             appeal.append("夫妻共同债务")
    #         if re.search("同居",text):
    #             appeal.append("同居问题")
    #         if re.search("继承",text):
    #             appeal.append("遗产继承")
    #         if re.search("抚养费",text):
    #             appeal.append("支付抚养费")
    #         if re.search("抚养权",text):
    #             appeal.append("确认抚养权")
    #         if re.search("赡养",text):
    #             appeal.append("支付赡养费")
    #         if re.search("彩礼",text):
    #             appeal.append("返还彩礼")
    #         if len(appeal) < 1:
    #             appeal = "离婚"
    #         return appeal
    #     elif model_name == "劳动劳务":
    #         if re.search("工伤",text):
    #             appeal.append("工伤赔偿")
    #         if re.search("社保",text):
    #             appeal.append("社保待遇")
    #         if re.search("工资|报酬",text):
    #             appeal.append("支付劳动劳务报酬")
    #         if len(appeal) < 1:
    #             appeal = ""
    #         return appeal
    #     elif model_name == "银行借贷":
    #         if re.search("有效|算",text):
    #             appeal.append("借贷关系")
    #         if re.search("利息|本金",text):
    #             appeal.append("还本付息")
    #         if len(appeal) < 1:
    #             appeal = ""
    #         return appeal
    #     elif model_name == "交通事故":
    #         if re.search("赔偿|医药费|误工费",text):
    #             appeal.append("损害赔偿")
    #         if re.search("扣.*分",text):
    #             appeal.append("违章扣分")
    #         if len(appeal) < 1:
    #             appeal = "损害赔偿"
    #         return appeal
    #     elif model_name == "合同纠纷":
    #         if re.search("买卖",text):
    #             appeal.append("买卖合同")
    #         if len(appeal) < 1:
    #             appeal = ""
    #         return appeal
    #     elif model_name == "房产物业":
    #         if re.search("房",text):
    #             appeal.append("房产纠纷")
    #         if re.search("物业",text):
    #             appeal.append("物业服务")
    #         if re.search("装修",text):
    #             appeal.append("装饰装修纠纷")
    #         if re.search("邻",text):
    #             appeal.append("相邻侵害问题")
    #         if len(appeal) < 1:
    #             appeal = ["房产纠纷"]
    #         return appeal
    #     elif model_name == "侵权纠纷":
    #         if re.search("物权",text):
    #             appeal.append("物权保护")
    #         if re.search("隐私",text):
    #             appeal.append("隐私权纠纷")
    #         if re.search("人格",text):
    #             appeal.append("人格权纠纷")
    #         if re.search("抵押",text):
    #             appeal.append("抵押纠纷")
    #         if len(appeal) < 1:
    #             appeal = ""
    #         return appeal
    #     elif model_name == "建设工程":
    #         if re.search("施工合同",text):
    #             appeal.append("施工合同")
    #         if re.search("建设用地",text):
    #             appeal.append("建设用地合同")
    #         if re.search("承包",text):
    #             appeal.append("承包经营权")
    #         if re.search("土地.*承包",text):
    #             appeal.append("土地承包合同")
    #         if len(appeal) < 1:
    #             appeal = ""
    #         return appeal
    #     elif model_name == "知识产权":
    #         if re.search("著作",text):
    #             appeal.append("确认著作权归属")
    #             appeal.append("著作权侵权赔偿")
    #         if re.search("商标",text):
    #             appeal.append("确认商标权归属")
    #             appeal.append("商标权侵权赔偿")
    #         if re.search("专利",text):
    #             appeal.append("确认专利权归属")
    #             appeal.append("专利权侵权赔偿")
    #         if re.search("竞争",text):
    #             appeal.append("不正当竞争损害赔偿")
    #         if len(appeal) < 1:
    #             appeal = ""
    #         return appeal
    #     else:
    #         return ""
    ask_type = ""
    appeal = []
    df_ = suqiu_keywords[suqiu_keywords["ask_type"] == model_name]
    for index, row in df_.iterrows():
        problem = row["problem"]
        if problem == "继承问题":
            continue
        suqiu = row["suqiu"]
        positive_keywords = row["positive_keywords"]
        negative_keywords = row["negative_keywords"]
        if ask_type != "" and ask_type != problem:
            continue
        if str(positive_keywords) != "nan":
            if re.search(positive_keywords, text):
                if str(negative_keywords) != "nan":
                    if not re.search(negative_keywords, text):
                        ask_type = problem
                        appeal.append(suqiu)
                else:
                    ask_type = problem
                    appeal.append(suqiu)
    if ask_type == "":
        ask_type = model_name
    return ask_type.lstrip("企业"), appeal


if __name__ == "__main__":
    text = "而且我的是一个多月的新车，要大修需要一个多月，可以索赔折旧费和误工费代步费吗？"
    _, label, prob = get_question_feature_label_prob(text)
    print(label, prob)
    ask_type, appeal = get_appeal_by_rules(text, label)
    print(ask_type, appeal)
