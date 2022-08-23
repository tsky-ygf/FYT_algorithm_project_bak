#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/20 15:43
# @Author  : Adolf
# @Site    : 
# @File    : theft_extraction.py
# @Software: PyCharm
from paddlenlp import Taskflow
from pprint import pprint

use_schema = {
    '盗窃触发词': [
        '总金额',
        '物品',
        '地点',
        '时间',
        '人物',
        '行为'
    ]
}
ie = Taskflow('information_extraction', schema=use_schema, device_id=3,
              task_path='model/uie_model/criminal/theft/model_best/')


def get_theft_result(text):
    res_event = ie(text)
    # print(res_event)
    return res_event


# text = "现已审理终结。浙江省诸暨市人民检察院指控，2019年7月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在" \
#        "3楼21号病床床头柜内窃得被害人俞某的皮包一只，内有现金￥1500元和银行卡、身份证等财物。"

text = "法院审理查明：\n2012年10月25日1时许，被告人郑小明伙同彭小清窜到诸暨市暨阳街道跨湖路99号永鑫花园，撬断围栏进入小区，由被告人郑小" \
       "明望风，彭小清通过下水道进入小区20幢5单元被害人杨燕家，窃得部分财物，后两人翻墙逃跑。被告人郑小明在逃跑途中被民警抓获。\n被告人郑小明" \
       "于2009年5月21日因犯盗窃罪被江西省抚州市临川区人民法院判处有期徒刑4年6个月，于2012年9月7日刑满释放。"

rest = get_theft_result(text)
pprint(rest)
