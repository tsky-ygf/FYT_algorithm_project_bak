#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/09 11:25
# @Author  : Czq
# @File    : bak.py
# @Software: PyCharm

import pandas as pd

schema_df = pd.read_csv('DocumentReview/Config/config_common.csv')
schemas = schema_df['schema'].values
common2alias_dict = dict()
for cont_type in ['baomi']:
    columns = schema_df[cont_type].values
    common2alias = dict()
    for sche, alias in zip(schemas, columns):
        if sche in ['争议解决', '通知与送达', '甲方解除合同', '乙方解除合同', '未尽事宜', '金额']:
            continue
        sche = sche.strip()
        alias = alias.strip()
        common2alias[sche] = alias
    common2alias_dict[cont_type] = common2alias

schemas = schemas.tolist()
schemas.remove('争议解决')
schemas.remove('通知与送达')
schemas.remove('甲方解除合同')
schemas.remove('乙方解除合同')
schemas.remove('未尽事宜')
schemas.remove('金额')

print(schemas, common2alias_dict)