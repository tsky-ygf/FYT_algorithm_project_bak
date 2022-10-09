#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/09 11:25
# @Author  : Czq
# @File    : bak.py
# @Software: PyCharm

import pandas as pd

schema_df = pd.read_csv('DocumentReview/Config/config_common.csv')
schemas = schema_df['schema'].values
common2alias = dict()
columns = schema_df.columns
for sche in schemas:
#     common2alias[sche] = dict()
#     for col in columns:
#         common2alias[sche][col] = None
        pass