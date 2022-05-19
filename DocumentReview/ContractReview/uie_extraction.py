#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 09:16
# @Author  : Adolf
# @Site    : 
# @File    : uie_extraction.py
# @Software: PyCharm
import re
from DocumentReview.ParseFile.parse_word import read_docx_file
from pprint import pprint

text_list = read_docx_file(docx_path="data/DocData/LaborContract/劳动合同.docx")