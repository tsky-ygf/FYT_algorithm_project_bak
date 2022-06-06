#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 13:20
# @Author  : Adolf
# @Site    : 
# @File    : google.py
# @Software: PyCharm
# import re
# import html
# from urllib import parse
# import requests
from googletrans import Translator

# GOOGLE_TRANSLATE_URL = 'http://translate.google.cn/m?q=%s&tl=%s&sl=%s'
#
#
# def translate(text, to_language="auto", text_language="auto"):
#     text = parse.quote(text)
#     url = GOOGLE_TRANSLATE_URL % (text, to_language, text_language)
#     response = requests.get(url)
#     data = response.text
#     expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
#     result = re.findall(expr, data)
#     if (len(result) == 0):
#         return ""

#     return html.unescape(result[0])
#
#
# print(translate("你吃饭了么?", "en", "zh-CN"))  # 汉语转英语
# print(translate("你吃饭了么？", "ja", "zh-CN"))  # 汉语转日语
# print(translate("about your situation", "zh-CN", "en"))  # 英语转汉语

# 设置Google翻译服务地址
translator = Translator(service_urls=[
    'translate.google.cn'
])

translation = translator.translate('hello world', dest='zh-CN')
print(translation.text)
