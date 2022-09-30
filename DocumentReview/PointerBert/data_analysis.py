#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/29 15:18
# @Author  : Czq
# @File    : data_analysis.py
# @Software: PyCharm
import json
import os
from collections import defaultdict
from pprint import pprint

import numpy

from DocumentReview.PointerBert.utils import read_config_to_label


def fun1():
    file = 'data/data_src/common_long/train.json'
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            # if i == 3:
                j = json.loads(line)
                text = j['text']
                entities = j['entities']
                print(text)
                print(j['id'])
                print('-'*100)

                for entity in entities:
                    if entity['start_offset'] is None:
                        print(entity)
                    # print(entity['label'], ':::', text[entity['start_offset']:entity['end_offset']])
                    # print(entity)
                        print('*'*50)


def fun2():
    labels2id, alias2label = read_config_to_label(None)

    print("label number", len(labels2id))
    file_path = 'data/data_src/common_0926'
    data_all = []
    to_file = 'data/data_src/common_all/common_all.json'
    w = open(to_file, 'w', encoding='utf-8')
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join('data/data_src/common_0926', file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                if 'caigou' in file or 'common.jsonl' in file or 'jietiao' in file:
                    labels = line['entities']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab['label'] not in alias2label:
                            continue
                        label = alias2label[lab['label']]
                        if label not in labels2id:
                            continue
                        new_labels[lab['label']].append([lab['start_offset'], lab['end_offset']])
                    for lab, indexs in new_labels.items():
                        if len(indexs) == 1:
                            continue
                        indexs.sort()
                        for ii in range(1, len(indexs)):
                            if indexs[ii][0] == indexs[ii][1]:
                                print("there are", new_labels)
                                print("id:", idd)
                                print("text ", text)

                else:
                    labels = line['label']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab[2] not in alias2label:
                            continue
                        label = alias2label[lab[2]]
                        if label not in labels2id:
                            continue
                        new_labels[label].append([lab[0],lab[1]])

                    for lab, indexs in new_labels.items():
                        if len(indexs) == 1:
                            continue
                        indexs.sort()
                        for ii in range(1, len(indexs)):
                            if indexs[ii-1][1] == indexs[ii][0]:
                                print("there are", new_labels)
                                print("id:", idd)
                                print("text ", text)


def fun3():
    labels2id, alias2label = read_config_to_label(None)
    labels2id.append('争议解决')
    labels2id.append('通知与送达')
    labels2id.append('未尽事宜')
    labels2id.append('附件')
    print("label number", len(labels2id))
    file_path = 'data/data_src/common_0926'
    for file in os.listdir(file_path):
        print(file)
        file = os.path.join('data/data_src/common_0926', file)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                idd = line['id']
                text = line['text'].replace('\xa0', ' ')
                dp = numpy.zeros(len(text))
                if 'caigou' in file or 'common.jsonl' in file or 'jietiao' in file:
                    labels = line['entities']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab['label'] not in alias2label:
                            continue
                        label = alias2label[lab['label']]
                        if label not in labels2id:
                            continue
                        new_labels[lab['label']].append([lab['start_offset'], lab['end_offset']])
                        dp[lab['start_offset']:lab['end_offset']] +=1
                else:
                    labels = line['label']
                    new_labels = defaultdict(list)
                    for lab in labels:
                        if lab[2] not in alias2label:
                            continue
                        label = alias2label[lab[2]]
                        if label not in labels2id:
                            continue
                        new_labels[lab[2]].append([lab[0],lab[1]])
                        dp[lab[0]:lab[1]] += 1
                if 2 in dp:
                    print(idd, text, new_labels)
                    for l, v in new_labels.items():
                        for _ in v:
                            print(l, text[_[0]:_[1]])





if __name__ == "__main__":
    # print("wefwefwfe"[1:4])
    # print("wefwefwfe"[1:None]) # efwefwfe
    # fun1()  # 查看标注
    # fun2()  # 验证连续的标签
    fun3() # 检查重叠
    # from transformers import AutoTokenizer
    # t = AutoTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')
    # print(len("今天天气是阴天，明天是多云，后天是晴天"))
    #     # res = t("今天天气是阴天，明天是多云，后天是晴天",
    #     #         truncation=True,
    #     #         max_length=12,
    #     #         stride=5,
    #     #         return_overflowing_tokens=True,
    #     #         return_offsets_mapping=True,
    #     #         padding="max_length")
    # print(len(res['input_ids'][0]), len(res['input_ids'][1]),len(res['input_ids'][2]))
    # print(res)
    text = '劳动合同甲方（用人单位）名称：毕博诚传媒有限公司住所：天津市伟市上街赵路A座996321号法定代表人（或主要负责人）:徐秀荣乙方（劳动者）姓名：李丹性别：女身份证号码：231201196301101322通讯地址:吉林省丽市萧山黎路E座175121号联系电话：15627312815根据《中华人民共和国劳动法》、《中华人民共和国劳动合同法》等法律、法规、规章的规定，在平等自愿，协商一致的基础上，同意订立本劳动合同，共同遵守本合同所列条款。第一条劳动合同类型及期限一、劳动合同类型及期限按下列第1项确定。1、固定期限：自2021年3月12日起至2022年3月11日止。2、无固定期限：自年月日起至法定的解除或终止合同的条件出现时止。3、以完成一定工作为期限：。二、本合同约定试用期，试用期自2021年3月12日起至2021年4月11日止。第二条工作内容、工作地点及要求1、乙方从事厨师工作,工作地点在天津。2、乙方工作应达到以下标准3、根据甲方工作需要，经甲、乙双方协商同意，可以变更工作岗位、工作地点。第三条工作时间和休息休假一、工作时间按下列第2项确定：1、实行标准工时制。2、实行经劳动保障行政部门批准实行的不定时工作制。3、实行经劳动保障行政部门批准实行的综合计算工时工作制。结算周期：按/结算。二、甲方由于生产经营需要经与工会和乙方协商后可以延长乙方工作时间，一般每日不得超过4小时，甲方依法保证乙方的休息休假权利。第四条劳动报酬及支付方式与时间一、乙方试用期间的月劳动报酬为4000元。二、试用期满后，乙方在法定工作时间内提供正常劳动的月劳动报酬为6000元。乙方工资的增减，奖金、津贴、补贴、加班加点工资的发放，以及特殊情况下的工资支付等，均按相关法律法规及甲方依法制定的规章制度执行。甲方支付给乙方的工资不得低于当地最低工资标准。三、甲方的工资发放日为每月20日。甲方应当以货币形式按月支付工资，不得拖欠。四、乙方在享受法定休假日以及依法参加社会活动期间，甲方应当依法支付工资。第五条社会保险甲、乙双方必须依法参加社会保险，按月缴纳社会保险费。乙方缴纳部分，由甲方在乙方工资中代为扣缴。第六条劳动保护、劳动条件和职业危害防护甲乙双方都必须严格执行国家有关安全生产、劳动保护、职业卫生等规定。有职业危害的工种应在合同约定中告知，甲方应为乙方的生产工作提供符合规定的劳动保护设施、劳动防护用品及其他劳动保护条件。乙方应严格遵守各项安全操作规程。甲方必须自觉执行国家有关女职工劳动保护和未成年工特殊保护规定。第七条劳动合同变更、解除、终止一、经甲乙双方协商一致，可以变更劳动合同相关内容。变更劳动合同，应当采用书面形式。变更后的劳动合同文本由甲乙双方各执一份。二、经甲乙双方协商一致，可以解除劳动合同。三、乙方提前三十日以书面形式通知甲方，可以解除劳动合同。乙方在试用期内提前三日通知甲方，可以解除劳动合同。四、甲方有下列情形之一的，乙方可以解除劳动合同：1、未按劳动合同约定提供劳动保护或者劳动条件的；2、未及时足额支付劳动报酬的；3、未依法缴纳社会保险费的；4、规章制度违反法律、法规的规定，损害乙方权益的；5、以欺诈、胁迫的手段或乘人之危，使乙方在违背真实意思的情况下订立或者变更劳动合同致使劳动合同无效的；6、法律、法规规定乙方可以解除劳动合同的其他情形。甲方以暴力、威胁或者非法限制人身自由的手段强迫乙方劳动的，或者甲方违章指挥、强令冒险作业危及乙方人身安全的，乙方可以立即解除劳动合同，不需事先告知甲方。五、乙方具有下列情形之一的，甲方可以解除本合同：1、在试用期间被证明不符合录用条件的；2、严重违反甲方的规章制度的；3、严重失职、营私舞弊，给甲方造成重大损害的；4、同时与其他用人单位建立劳动关系，对完成甲方的工作任务造成严重影响，或者经甲方提出，拒不改正的。5、以欺诈、胁迫的手段或乘人之危，使甲方在违背真实意思的情况下订立或者变更劳动合同致使劳动合同无效的。6、被依法追究刑事责任的。六、下列情形之一，甲方提前三十日以书面形式通知乙方后，可以解除本合同：1、乙方患病或者非因工负伤，在规定的医疗期满后不能从事原工作，也不能从事由甲方另行安排的工作的；2、乙方不能胜任工作，经过培训或者调整工作岗位，仍不能胜任工作的；3、劳动合同订立时所依据的客观情况发生重大变化，致使原劳动合同无法履行，经甲乙双方协商，不能就变更劳动合同内容达成协议的。七、甲方依照企业破产法规定进行重整的；或生产经营发生严重困难的；或企业转产、重大技术革新或者经营方式调整，经变更劳动合同后，仍需裁减人员的；或其他因劳动合同订立时所依据的客观经济情况发生重大变化，致使劳动合同无法履行的,应当提前三十日向工会或者全体职工说明情况，听取工会或者职工意见，裁减人员方案以书面形式向劳动行政部门报告后，可以解除劳动合同。八、有下列情形之一的,劳动合同终止:1、劳动合同期满的；2、乙方开始依法享受基本养老保险待遇的;3、乙方死亡,或者被人民法院宣告死亡或者宣告失踪的;4、甲方被依法宣告破产，被吊销营业执照、责令关闭、撤销或者甲方决定提前解散的；5、法律、行政法规规定的其他情形。九、劳动合同期满,乙方具有下列情形之一的,劳动合同应当续延至相应的情形消失时终止；1、从事接触职业病危害作业的劳动者未进行离岗前职业健康检查，或者疑似职业病病人在诊断或者医学观察期间的；2、在本单位患职业病或者因工负伤被确认丧失或者部分丧失劳动'
    print(text[616:686])
    print(text.index('6000元'))
    pass