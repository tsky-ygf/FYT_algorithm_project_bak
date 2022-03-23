#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:04
# @Author  : Adolf
# @Site    : 
# @File    : parse_original_data.py
# @Software: PyCharm
import pandas as pd
from docx import Document
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

law_config = pd.read_csv("")