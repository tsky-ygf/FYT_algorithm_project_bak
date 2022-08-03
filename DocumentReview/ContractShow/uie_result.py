#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 10:16
# @Author  : Adolf
# @Site    : 
# @File    : uie_result.py
# @Software: PyCharm
import streamlit as st
import pandas as pd
from paddlenlp import Taskflow
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

contract_type = st.sidebar.selectbox("请选择合同类型", ["房屋租赁", "劳动雇佣", "借款合同", '劳动合同', '一般买卖', '欠条'], key="合同类型")
if contract_type == '房屋租赁':
    config_path = 'DocumentReview/Config_bak/LeaseConfig/fangwu.csv'
    model_path = 'model/uie_model/fwzl/model_best'

elif contract_type == '劳动雇佣':
    config_path = 'DocumentReview/Config_bak/LaborConfig/laowu.csv'
    model_path = 'model/uie_model/guyong/model_best'

elif contract_type == '借款合同':
    config_path = 'DocumentReview/Config_bak/LoanConfig/jiekuan_20220615.csv'
    model_path = 'model/uie_model/jkht/model_best'

elif contract_type == '劳动合同':
    config_path = 'DocumentReview/Config_bak/LaborConfig/labor_20220615.csv'
    model_path = 'model/uie_model/laodong/model_best'

elif contract_type == '一般买卖':
    config_path = 'DocumentReview/Config_bak/BusinessConfig/maimai.csv'
    model_path = 'model/uie_model/maimai/model_best'

elif contract_type == '欠条':
    config_path = 'DocumentReview/Config_bak/LoanConfig/jietiao_20220531.csv'
    model_path = 'model/uie_model/model_best'

else:
    raise ValueError('请选择合同类型')

config = pd.read_csv(config_path)
# print(maimai_config)
# model_path = 'model/uie_model/fwzl/model_best'

use_schema = config.schema.tolist()
# print(maimai_schema)

text = st.text_area(label="请输入文本内容", height=300, value="""个人简单房屋租赁合同出租方∶刘恺威（以下简称甲方）
承租方∶杨幂（以下简称乙方）
甲、乙双方就房屋租赁事宜，达成如下协议∶
一、甲方将位于杭州海威新界2304号出租给乙方居住使用，租赁期限自2022年1月1日至 2027年1月1日止，计60个月。
二、本房屋年租金为人民币200000万元，按年结算。已付定金，其余的在2022年2月1日一次性付清。
三、乙方租赁期间，水费、电费、以及其他由乙方居住而产生的费用由乙方负担。租赁结束时，乙方须缴清欠费。
四.房屋租赁期为五年，从2022年1月1日至 2027年1月1日。在此期间.任何一方要求终止合同，须提前二个月通知对方，并偿付对方总租金50%的违约金；
五、在承租期间，未经甲方同意，乙方无权转租或转借该房屋；不得改变房屋结构及其用途，由于乙方人为原因造成该房屋及其配套设施损坏的，由乙方承担赔偿素任。
六、就本合同发生纠纷，双方协商解决，协商不成，任何一方均有权向杭州人民法院提起诉讼，请求司法解决。
七、本合同连一式2份，甲、乙双方各执1份，自双方签字之日起生效。
甲方签字∶
乙方签字∶
2022年1月1日""", key="text")
run = st.button("预测")
if run:
    ie = Taskflow('information_extraction', schema=use_schema, device_id=2, task_path=model_path)
    res = ie(text)
    # print(res)

    find_schema = []
    for key, values in res[0].items():
        st.markdown('### 审核要点:{}'.format(key))
        for index, value in enumerate(values):
            st.write(index + 1, '.', value['text'])
        find_schema.append(key)
        # break
    st.markdown('### 未抽取到的审核要点')
    print(use_schema)
    for con in use_schema:
        if con not in find_schema:
            st.write('- {}'.format(con))
