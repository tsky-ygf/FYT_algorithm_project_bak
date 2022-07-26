# coding=utf-8
import fasttext
import pandas as pd
import jieba
import re
import traceback
import numpy as np
import os
import sys
from LawsuitPrejudgment.common import prob_ps_desc


model = fasttext.load_model('../model/ft/ft.bin')


def predict(problem, suqiu, fact):
    try:
        fact = prob_ps_desc[problem+'_'+suqiu] + '。' + fact
        fact = ' '.join(jieba.lcut(fact))
        prediction = model.predict_proba([fact])[0][0]
        label = int(prediction[0].replace('__label__', ''))
        proba = prediction[1]
        if label==0:
            proba = 1 - proba
        return proba
    except:
        traceback.print_exc()
        return None


if __name__=='__main__':
    print(predict('婚姻继承', '离婚', '我老公经常打我'))
