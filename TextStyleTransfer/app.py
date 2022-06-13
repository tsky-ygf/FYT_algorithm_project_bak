#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 13:27
# @Author  : Adolf
# @Site    : 
# @File    : app.py
# @Software: PyCharm
import traceback
import logging
import logging.handlers
from flask import Flask, request
from flask_cors import CORS
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

tokenizer1 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model1 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

tokenizer2 = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model2 = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

translation1 = pipeline(task='translation', model=model1, tokenizer=tokenizer1, device=3)
translation2 = pipeline(task='translation', model=model2, tokenizer=tokenizer2, device=3)


@app.route('/translation', methods=["post"])
def get_translation_res():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            text = in_dict['content']
            # print(text)
            en_res = translation1(text)[0]['translation_text']
            ch_res = translation2(en_res)[0]['translation_text']
            return json.dumps({'result': ch_res, "status": 0}, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7999, debug=False)  # , use_reloader=False)
    # en_res = translation1('你好世界')[0]['translation_text']
    # ch_res = translation2(en_res)[0]['translation_text']
    # print(ch_res)