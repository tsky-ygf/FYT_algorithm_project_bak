#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 12:01
# @Author  : Adolf
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import os
import pandas as pd
from IntelligentConsultation.src.faq_pipeline.init_faq_tools import init_retriever, init_document_store
from IntelligentConsultation.src.faq_pipeline.core_model.common_model import MODEL_REGISTRY
from loguru import logger
from IntelligentConsultation.src.faq_pipeline.infer import FAQPredict

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def insert_data(data_path="data/fyt_train_use_data/QA/pro_qa.csv",
                index_name="topic_qa_test_v2",
                model_name="model/similarity_model/simcse-model-top-32"):
    document_store = init_document_store(index_name=index_name)
    retriever = init_retriever(document_store, model_name=model_name)

    logger.info("start insert data")

    df = pd.read_csv(data_path)
    df.fillna("0", inplace=True)
    questions = list(df["question"].values)
    df["query_emb"] = retriever.embed_queries(texts=questions)
    df = df.rename(columns={"question": "content"})
    docs_to_index = df.to_dict(orient="records")

    docs_to_index = [one for one in docs_to_index if len(one["answer"]) < 36000]
    document_store.write_documents(docs_to_index)
    # document_store.update_embeddings(retriever)
    logger.success("insert data success")


def test_acc(index_name, model_name):
    faq = FAQPredict(level="warning", index_name=index_name, model_name=model_name)

    test_data_path = "data/fyt_train_use_data/QA/test_qa.csv"
    df = pd.read_csv(test_data_path)

    acc_count = 0

    test_question_list = []
    correct_query_list = []
    result_query_list = []
    is_correct = []

    for index, row in df.iterrows():
        answer, similarity_question = faq(row["测试问题"])
        result = similarity_question[0]["question"]
        # break
        test_question_list.append(row["测试问题"])
        correct_query_list.append(row["query"])
        result_query_list.append(result)

        query_list = row["query"].split("|")

        if result == row["query"] or result in query_list:
            # print("right")
            acc_count += 1
            is_correct.append(1)
        else:
            logger.warning("question: {}".format(row["测试问题"]))

            logger.warning("正确答案:{}".format(row["query"]))
            logger.warning("result:{}".format(result))

            is_correct.append(0)
        # break

    test_result = {
        "测试问题": test_question_list,
        "正确问题": correct_query_list,
        "返回问题": result_query_list,
        "是否正确": is_correct,
    }

    test_result_df = pd.DataFrame(test_result)
    # test_result_df.to_csv("Consult/test_result.csv", index=False)
    accuracy = acc_count / df.shape[0]

    return accuracy, test_result_df


def faq_main(config):
    model = MODEL_REGISTRY.get(config["model_type"])(config["train_config"])
    model.train()

    insert_data(data_path=config["train_config"]["train_data_path"],
                index_name=config["index_name"],
                model_name=config["train_config"]["model_output_path"])

    acc, res_df = test_acc(index_name=config["index_name"],
                           model_name=config["train_config"]["model_output_path"])

    return acc, res_df


def param_optim(config):
    best_acc = 0
    for lr in [1e-5, 2e-5]:
        for train_batch_size in [32, 64, 128]:
            for max_seq_length in [64, 128]:
                for num_epochs in [1, 2, 3, 4]:

                    print("lr: {} ===> train_batch_size: {} ===> max_seq_length: {} ===> num_epochs: {}".
                          format(lr, train_batch_size, max_seq_length, num_epochs))

                    config["train_config"]["lr"] = lr
                    config['train_config']['train_batch_size'] = train_batch_size
                    config['train_config']['max_seq_length'] = max_seq_length
                    config['train_config']['num_epochs'] = num_epochs

                    acc, res_df = faq_main(config=config)
                    if acc > best_acc:
                        best_acc = acc
                        best_lr = lr
                        best_train_batch_size = train_batch_size
                        best_max_seq_length = max_seq_length
                        best_num_epochs = num_epochs

                        res_df.to_csv("IntelligentConsultation/src/faq_pipeline/result.csv", index=False)
                        print("acc:{}".format(acc))

                    print("best accuracy: {}".format(best_acc))
                    print("best lr: {}".format(best_lr))
                    print("best train_batch_size: {}".format(best_train_batch_size))
                    print("best max_seq_length: {}".format(best_max_seq_length))
                    print("best num_epochs: {}".format(best_num_epochs))


if __name__ == '__main__':
    # use_config = {
    #     "model_type": "SimCSE",
    #     "index_name": "topic_qa_test",
    #     "train_config": {
    #         "model_name": "model/language_model/chinese-roberta-wwm-ext",
    #         "model_output_path": "model/similarity_model/simcse-model-top-32",
    #         "train_data_path": "data/fyt_train_use_data/QA/pro_qa.csv",
    #         "lr": 5e-5,
    #         "train_batch_size": 128,
    #         "max_seq_length": 32,
    #         "num_epochs": 1}
    # }
    # param_optim(use_config)
    #
    insert_data(data_path="data/fyt_train_use_data/QA/origin_data.csv",
                model_name="model/similarity_model/simcse-model-all",
                index_name="topic_qa_test")
    # acc, res_df = test_acc(index_name="topic_qa_test", model_name="model/similarity_model/simcse-model-topic-qa")
    # res_df.to_csv("data/fyt_train_use_data/QA/pro_qa_res.csv", index=False)
    # print(acc)
