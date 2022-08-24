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


text = "被告人唐志强，男，1959年11月15日出生于上海市青浦区，汉族，文盲，无业，住上海市青浦区。因盗窃于2013年9月被龙游县公安局" \
       "行政拘留十四日。因犯盗窃罪于2015年8月被绍兴市柯桥区人民法院判处拘役五个月，并处罚金一千元。因犯盗窃罪于2016年12月被慈溪市人民" \
       "法院判处有期徒刑六个月，并处罚金一千元。因犯盗窃罪于2018年1月被绍兴市越城区人民法院判处有期徒刑九个月，并处罚金一千元。因犯盗" \
       "窃罪于2018年9月29日被宁波市鄞州区人民法院判处有期徒刑十个月，并处罚金一千元，于2019年4月19日刑满释放。因本案于2019年7月23日" \
       "被诸暨市公安局刑事拘留，同年8月3日被逮捕。现羁押于诸暨市看守所。浙江省诸暨市人民检察院以诸检刑诉（2019）1190号起诉书" \
       "指控被告人唐志强犯盗窃罪，于2019年10月9日向本院提起公诉。本院于同次日立案，并依法适用简易程序，实行独任审判，公开开庭审理了本" \
       "案。浙江省诸暨市人民检察院指派检察员杨瑞霞出庭支持公诉，被告人唐志强到庭参加诉讼。现已审理终结。浙江省诸暨市人民检察院指控，2019年7" \
       "月22日10时30分许，被告人唐志强窜至诸暨市妇幼保健医院，在3楼21号病床床头柜内窃得被害人俞某的皮包一只，内有现金￥1500元" \
       "和银行卡、身份证等财物。"

rest = get_theft_result(text)
pprint(rest)

