from keras.models import load_model
import os
import pickle
import jieba
import numpy as np

problem_suqius = {
    '婚姻家庭': ['离婚', '返还彩礼']
}
base_path = '../model/checkpoint/'


def deep_predict(input_text, problem, suqiu, maxlen=180, model_type='textcnn', threshold=0.5):
    if problem not in problem_suqius:
        return None
    if suqiu not in problem_suqius[problem]:
        return None

    model_ckp = suqiu + '_' + model_type + '_model.h5'
    word2index_path = suqiu + '_word2index.pk'
    model_path = os.path.join(base_path, model_ckp)
    word2index_path = os.path.join(base_path, word2index_path)
    with open(word2index_path, 'rb') as f:
        word2index = pickle.load(f)
    cut_words = list(jieba.cut(input_text))

    # index2word = {v: k for k, v in word2index.items()}

    seq = [word2index[word] for word in cut_words if word in word2index]
    if len(seq) > maxlen:
        seq = seq[:maxlen]
    else:
        seq.extend([0] * (maxlen - len(seq)))

    inputs = np.array([seq])
    model = load_model(model_path)
    result = model.predict(inputs)[0][0]
    return result


if __name__ == "__main__":
    print(deep_predict('老公经常赌博，虐待家人，一审不让我们离婚，现在感情已经彻底破裂了', '婚姻家庭', '离婚'))
