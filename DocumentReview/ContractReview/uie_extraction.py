#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/19 09:16
# @Author  : Adolf
# @Site    : 
# @File    : uie_extraction.py
# @Software: PyCharm
from DocumentReview.ParseFile.parse_word import read_docx_file
from pprint import pprint
from paddlenlp import Taskflow

text = '''执法部门:【浙江省杭州市临安区】浙江省杭州市临安区综合行政执法局
行政处罚决定书:经查明，当事人楼国军于2022年04月24日驾驶车牌号为***A6J8C6的蓝
色轻型自卸货车在杭州市*********的项目工地内运载工程渣土（碎石）到杭州市*********
公墓西侧的场地内进行倾倒，共倾倒五车，于2022年04月25日11时18分被本局通过监控
巡查发现。经查实，车牌号为***A6J8C6的蓝色轻型自卸货车装运工程渣土（碎石）未取
得相关合法有效的准运手续。当事人的行为属于无准运证运输工程渣土，违反《杭州市城
市市容和环境卫生管理条例》第六十一条第一款之规定。执法队员于2022年04月28日向
当事人送达了《责令立即改正违法行为通知书》（临综执责改字〔2022〕第0002503号
），当事人及时改正了违法行为。另经查证，当事人为本年度第一次从事该性质的违法行
为，源头为区内且存在倾倒情形。现根据《杭州市城市市容和环境卫生管理条例》第六十
一条第二款之规定，结合《杭州市城市管理行政处罚自由裁量权实施办法》，经本局负责
人批准，决定对当事人作出如下行政处罚：罚款人民币贰仟元整（￥2000）。'''

schema = ['行政主体', '行为', '尺度', '处罚', '证据', '法律依据', '违法行为']
ie = Taskflow('information_extraction', schema=schema, device_id=1, task_path="model/uie_model/xz2/model_best")
pprint(ie(text))
