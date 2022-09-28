import jieba
import jieba.posseg as jbpseg
import pandas as pd
import numpy as np
import pickle
import os
import joblib
#from scipy.spatial.distance import cosine
"""
加载词汇表、tfidf权重，计算句子相似性
"""
          # 名称,      地名 其它专名,形容词,名形词,b:区别词,d副词,f:方位词,g:语素,p:介词,s:处所词,v:动词,vn:名动词,vd:副动词
inc_pos = {'n', 'np', 'ns', 'nz', 'a', 'an', 'b', 'dg', 'd', 'f', 'g', 'p', 's', 'tg', 'vg', 'v', 'vn', 'vd', 'Ng', 'l', 'i', 'j', 'k', 'h','t'} # 't'

EMB_SZ=100
zero_emb = np.zeros(EMB_SZ)
def pseg_txt(inp):
    """
    过滤一些词，返回过滤并分词后的字符串。如：输入：'你好吗我的朋友'；输出：'你好 朋友'
    """
    "Parse word bag wrt predefined pos scope."
    if len(inp.strip()) == 0:
        return ''

    outp = []
    pseg_outp = jbpseg.cut(inp)
    for k,v in pseg_outp:
        if len(k) > 1 and v in inc_pos:
            outp.append(k)
    return ' '.join(outp)

def load_vocab_idf(vocab_file,idf_file):
    """
    得到词汇表和idf值
    :return:
    """
    # 1.词汇表
    vocab_object=open(vocab_file,'r')
    vocab_lines=[xx.strip() for xx in vocab_object.readlines()]
    # 2.idf信息
    idf_data=pd.read_csv(idf_file)
    idf_dict={}
    for index, row in idf_data.iterrows():
        word=row['word']
        idf=row['idf']
        if word in vocab_lines:
            idf_dict[word]=idf

    return idf_dict,vocab_lines

def compute_sentence_embedding(input_sentence,vocab_dict,vcoab_embedding,idfs_dict,tfidf_vectorizer,tfidf_vocab_dict):
    """
    通过结合tfidf和词向量，计算一个句子的向量表示
    :param inp_psegs:
    :param vocab_dict:
    :param vcoab_embedding:
    :param idfs_dict:
    :return:
    """
    input_sentence_filtter=pseg_txt(input_sentence) # input_sentence_filtter: "你好 吗 我 的 朋友"
    inp_psegs=input_sentence_filtter.split(" ")
    emb = np.zeros(EMB_SZ)
    print("emb:",emb)
    query_filtered=[]
    print("[input_sentence_filtter]:",input_sentence) # [input_sentence_filtter]
    ########################### query releted tfidf information ##################################
    input_query_matrix=tfidf_vectorizer.transform([input_sentence]) # [input_sentence_filtter]
    tfidf_score=list(input_query_matrix.toarray()[0])
    #print("####tfidf_score:",tfidf_score)
    word_score_dict = {}
    for i, word in enumerate(inp_psegs):
        index = tfidf_vocab_dict.get(word.lower(), None)
        #print('####@',i, ";word:", word,";index:",index)
        if index is not None:
            score = tfidf_score[index]
            #print('####@', i, ";word:", word, ";index:", index,";score:",score)
            word_score_dict[word] = score
    ########################## query releted tfidf information####################################
    for word in inp_psegs:
        if vocab_dict.get(word,None) is not None:
            query_filtered.append(word)
            #print("##word:",word,";tfidf score:",word_score_dict.get(word, 0.0)) # idfs_dict
            emb += vcoab_embedding.get(word, zero_emb) * word_score_dict.get(word, 0.0) # idfs_dict
    return emb,query_filtered

def cosine_similiarity(A,B):
    """
    余弦相似度
    :param A:
    :param B:
    :return:
    """
    num = np.sum(A * B.T) # 若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim

def _load_vocab_embedding_idf(data_path,vocab_file,vocab_embedding_file,idf_file):
    """
    加载词汇表、词汇向量、idf分数
    :return:
    """
    # 1.load vocab_list
    vocab_object=open(vocab_file,'r')
    vocab_data=pd.read_csv(vocab_object)
    vocab_list=list(vocab_data['vocab'])
    vocab_object.close()
    print("1.load_vocab_embedding_idf.length of vocab_list:",len(vocab_list))

    # 2. load idf dict
    idf_data=pd.read_csv(idf_file)
    idfs_dict={}
    vocab_dict={vocab:1 for vocab in vocab_list}
    for index,row in idf_data.iterrows():
        word=row['word']
        idf=float(row['idf'])
        if vocab_dict.get(word,None) is not None:
            idfs_dict[word]=idf
    print("2.length of idfs_dict:",len(idfs_dict))

    # 3.load vocab_embedding_dict
    vocab_embeddiding_object=open(vocab_embedding_file,'r')
    vocab_embedding_list=vocab_embeddiding_object.readlines()
    vocab_embeddiding_object.close()
    vocab_embedding_dict={}
    for i,line in enumerate(vocab_embedding_list):
        if i==0: continue
        string_list=line.strip().split(" ")
        word=string_list[0]
        if vocab_dict.get(word, None) is not None:
            embedding=[float(x) for x in string_list[1:]]
            embedding = np.array(embedding)
            vocab_embedding_dict[word]=embedding
    print("3.load_vocab_embedding_idf.length of vocab_embedding_dict:",len(vocab_embedding_dict))

    vocab_dict={vocab_list[i]:i for i in range(len(vocab_list))}

    tfidf_vectorizer = joblib.load(data_path + 'tfidf_vectorizer.pik')
    print("4.loaded tfidf_vectorizer")
    tfidf_vocab_dict=tfidf_vectorizer.vocabulary_
    return vocab_list,vocab_dict, vocab_embedding_dict, idfs_dict,tfidf_vectorizer,tfidf_vocab_dict

def load_vocab_embedding_idf(data_path,vocab_file,vocab_embedding_file,idf_file):
    """
    load or save vocab,embedding,idf files
    :return:
    """
    # 1. load if pickle file exist
    cache_file_name=data_path+'vocab_embedding_idf.pik'
    if os.path.exists(cache_file_name):
        with open(cache_file_name, 'rb') as data_f:
            return pickle.load(data_f)

    # 2. get and save to pickle if not exist
    vocab_list, vocab_dict,vocab_embedding_dict, idfs_dict,tfidf_vectorizer,tfidf_vocab_dict=_load_vocab_embedding_idf(data_path,vocab_file,vocab_embedding_file,idf_file)
    with open(cache_file_name, 'ab') as target_file:
        pickle.dump((vocab_list, vocab_dict,vocab_embedding_dict, idfs_dict,tfidf_vectorizer,tfidf_vocab_dict), target_file)
        return vocab_list, vocab_dict,vocab_embedding_dict, idfs_dict,tfidf_vectorizer,tfidf_vocab_dict

def as_num(x):  # format condidence_score
    #print("x:",x)
    y = '{:.3f}'.format(x)  # 5f表示保留5位小数点的float型
    return str(float(y))

A=np.array([0.3,0.2,0.5])
B=np.array([0.366,0.20,0.5])
cos=cosine_similiarity(A,B)
print("cos:",cos)
#cos2=cosine(A,B)
#print("cos2:",cos2)
