#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 16:26
# @Author  : Adolf
# @Site    : 
# @File    : base_faiss.py
# @Software: PyCharm
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from torch import Tensor
from Utils.logger import Logger

import torch
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from typing import List, Tuple, Union


class VectorSearch:
    def __init__(self, model_name_or_path: str,
                 device: str = None,
                 num_cells: int = 100,
                 num_cells_in_search: int = 10,
                 faiss_index_path: str = None,
                 no_faiss_index_path: str = None,
                 sentences_or_file_path: Union[str, List[str]] = None, ):
        self.logger = Logger(name="VectorSearch", level="INFO").logger
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False

        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        if sentences_or_file_path is not None:
            if isinstance(sentences_or_file_path, str):
                sentences = []
                with open(sentences_or_file_path, "r") as f:
                    self.logger.info("Loading sentences from {} ...".format(sentences_or_file_path))
                    for line in tqdm(f):
                        sentences.append(line.rstrip())
                sentences_or_file_path = sentences

        if faiss_index_path is not None:
            self.is_faiss_index = True
            self.index = {
                "index": faiss.read_index(faiss_index_path),
                "sentences": sentences_or_file_path,
            }
        if no_faiss_index_path is not None:
            self.is_faiss_index = False
            self.index = {
                "index": np.load(no_faiss_index_path),
                "sentences": sentences_or_file_path,
            }

    def encode(self, sentence: Union[str, List[str]],
               device: str = None,
               batch_size: int = 64,
               max_length: int = 128,
               normalize_to_unit: bool = True,
               return_numpy: bool = False, ) -> Union[ndarray, Tensor]:
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                outputs = self.model(**inputs, return_dict=True)
                embeddings = outputs.pooler_output

                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    def similarity(self, queries: Union[str, List[str]],
                   keys: Union[str, List[str], ndarray],
                   device: str = None) -> Union[float, ndarray]:
        query_vecs = self.encode(queries, device=device, return_numpy=True)

        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True)  # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        similarities = cosine_similarity(query_vecs, key_vecs)
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])

        return similarities

    def build_index(self, sentences_or_file_path: Union[str, List[str]],
                    use_faiss: bool = None,
                    faiss_fast: bool = False,
                    device: str = None,
                    batch_size: int = 64,
                    save_faiss_index_path: str = None,
                    save_no_faiss_index_path: str = None):
        # if use_faiss is None or use_faiss:
        # try:
        #     # import faiss
        #     assert hasattr(faiss, "IndexFlatIP")
        #     use_faiss = True
        # except Exception as e:
        #     self.logger.warning(f"Faiss not installed, use_faiss is set to False: {e}")
        #     self.logger.warning(
        #         "Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program "
        #         "continues with brute force search.")
        #     use_faiss = False

        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                self.logger.info("Loading sentences from {} ...".format(sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        self.logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size,
                                 normalize_to_unit=True, return_numpy=True)

        self.logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1],
                                           min(self.num_cells, len(sentences_or_file_path)))
            else:
                index = quantizer

            # 使用GPU加速faiss
            # if (self.device == "cuda" and device != "cpu") or device == "cuda":
            #     if hasattr(faiss, "StandardGpuResources"):
            #         self.logger.info("Use GPU-version faiss")
            #         res = faiss.StandardGpuResources()
            #         res.setTempMemory(20 * 1024 * 1024 * 1024)
            #         index = faiss.index_cpu_to_gpu(res, 0, index)
            #     else:
            #         self.logger.info("Use CPU-version faiss")
            # else:
            #     self.logger.info("Use CPU-version faiss")

            # 使用faiss_fast加速faiss
            # if faiss_fast:
            #     index.train(embeddings.astype(np.float32))

            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index

        if self.is_faiss_index and save_faiss_index_path is not None:
            faiss.write_index(index, save_faiss_index_path)
        if not self.is_faiss_index and save_no_faiss_index_path is not None:
            np.save(save_no_faiss_index_path, index)

        self.logger.info("Finished building index.")

    def add_to_index(self, sentences_or_file_path: Union[str, List[str]],
                     device: str = None,
                     batch_size: int = 64):

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                self.logger.info("Loading sentences from {} ...".format(sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        self.logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True,
                                 return_numpy=True)

        if self.is_faiss_index:
            self.index["index"].add(embeddings.astype(np.float32))
        else:
            self.index["index"] = np.concatenate((self.index["index"], embeddings))
        self.index["sentences"] += sentences_or_file_path
        self.logger.info("Finished")

    def search(self, queries: Union[str, List[str]],
               device: str = None,
               threshold: float = 0.6,
               top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:

        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device)
                    combined_results.append(results)
                return combined_results

            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            return [(self.index["sentences"][i], score) for i, score in id_and_score]
        else:
            query_vecs = self.encode(queries, device=device, normalize_to_unit=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)

            def pack_single_result(dist, _idx):
                return [(self.index["sentences"][indexs], sen) for indexs, sen in zip(_idx, dist) if
                        sen >= threshold]

            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])


if __name__ == "__main__":
    # example_sentences = [
    #     '一个动物正在咬着一个人的手指。',
    #     '一个女人正在阅读一本书。',
    #     '一个男人在车库里举重。',
    #     '一个男人正在谈小提琴。',
    #     '一个男的正在吃东西。',
    #     '一个男的正在弹钢琴。',
    #     '一只熊猫正在爬山。',
    #     '一个男人正在弹吉他。',
    #     '一个女人正在切肉。',
    #     '一位女士正在拍照。'
    # ]
    example_queries = [
        '一个男人正在放音乐。',
        '一个女人正在拍照。'
    ]

    model_name = "model/language_model/simbert-chinese-base"
    simbert = VectorSearch(model_name_or_path=model_name, faiss_index_path='BasicTask/SearchEngine/test.index',
                           sentences_or_file_path='BasicTask/SearchEngine/example_sentences.txt')

    results_ = simbert.search(example_queries)
    for i_, result_ in enumerate(results_):
        print("Retrieval results for query: {}".format(example_queries[i_]))
        for sentence_, score_ in result_:
            print("    {}  (cosine similarity: {:.4f})".format(sentence_, score_))

    # print("\n=========Calculate cosine similarities between queries and sentences============\n")
    # similarities_ = simbert.similarity(example_queries, example_sentences)
    # print(similarities_)

    # print("\n=========Naive brute force search============\n")
    # simbert.build_index(example_sentences, use_faiss=False)
    # results_ = simbert.search(example_queries)
    # for i_, result_ in enumerate(results_):
    #     print("Retrieval results for query: {}".format(example_queries[i_]))
    #     for sentence_, score_ in result_:
    #         print("    {}  (cosine similarity: {:.4f})".format(sentence_, score_))
    #     print("")
    #
    # print("\n=========Search with Faiss backend============\n")
    # simbert.build_index(example_sentences, use_faiss=True)
    # results_ = simbert.search(example_queries)
    # for i_, result_ in enumerate(results_):
    #     print("Retrieval results for query: {}".format(example_queries[i_]))
    #     for sentence_, score_ in result_:
    #         print("    {}  (cosine similarity: {:.4f})".format(sentence_, score_))
    #     print("")
