#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/25 16:39
# @Author  : Adolf
# @Site    : 
# @File    : predict.py
# @Software: PyCharm
import re
import operator

from loguru import logger
import torch
from transformers import BertTokenizer, BertForMaskedLM

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class MacbertCorrected:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained("model/language_model/macbert4csc-base-chinese")
        self.model = BertForMaskedLM.from_pretrained("model/language_model/macbert4csc-base-chinese")
        self.model.to(self.device)

    @staticmethod
    def get_errors(corrected_text, origin_text):#纠错细节
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    def __call__(self, ori_text):
        logger.debug(len(ori_text))
        text_length = 256
        if len(ori_text) > text_length:#分句子
            result_list = re.split("[。|\n]", ori_text)
            texts = result_list
        else:
            texts = [ori_text]

        logger.info(texts)
        result = []
        with torch.no_grad():
            outputs = self.model(**self.tokenizer(texts, padding=True, return_tensors='pt').to(self.device))

        for ids, text in zip(outputs.logits, texts):
            _text = self.tokenizer.decode(torch.argmax(ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = _text[:len(text)]

            corrected_text, details = self.get_errors(corrected_text, text)
            result.append((text, corrected_text, details))
        logger.debug(result)
        return corrected_text, details



if __name__ == '__main__':
    # textlist = ["今天新情很好", "你找到你最喜欢的工作，我也很高心。"]
    textlist = "车辆运输租赁合同3 出租⽅：天天租赁公司承租⽅：悟空有限公司 根据《中华⼈民共和国合同法》和交通部有关规章，为明确双⽅" \
               "的.权利和义务，经双⽅协商⼀致，签订汽车租赁合同，设定下列条款，共同遵守。 ⼀、出租⽅车辆基本情况车型：大众SUV，车牌号" \
               "：浙ADR0023，必须使⽤92号汽油（柴油）。租赁车辆的车况，以发车的双⽅签字确认的《租赁车辆交接单》为准。车况（附⾏车证" \
               "复印件）：2020年购买，⽆任何⼤修记录及安全⾏驶记录。 ⼆、租⽤期限⾃2022年1⽉1⽇起⾄2023年1⽉1⽇⽌，共计1年。需延长租" \
               "⽤期限双⽅另⾏协商，签订补充协议。 三、租车⽤途主要用于接送员工上下班。 四、租⾦及其结算⽅式租金2000元/月，押⾦2000" \
               "元，在租车时⼀次性以现金⽅式缴付给出租⽅，还车时，在车辆完整⽆损的情况下，凭押⾦收据与甲⽅结算，结算时间为银⾏营业时间内" \
               "。 五、出租⽅的权利义务（⼀）、不承担租赁车辆在租赁期间内所发⽣交通事故或其他事故造成的⼀切后果，包括有关部门的罚款等" \
               "。（⼆）、不承担租赁车辆于租赁期间内引发的第三者责任。（三）、依照法律法规的规定出租⽅应有的权利。（四）、向承租⽅提供" \
               "性能良好，证件齐全有效的车辆。（五）、在收到承租⽅租⾦及⾜额押⾦后，将所租车辆交付给承租⽅。（六）、如车辆属于⾮承租⽅使" \
               "⽤不当所产⽣的⽆法投⼊正常⾏驶时，出租⽅可向承租⽅提供临时替换车辆。 六、承租⽅的权利义务（⼀）、租赁期间拥有所租车辆的" \
               "使⽤权（⼆）、租赁期间应严格遵守国家各项法律法规，并承担由于违章，肇事，违法等⾏为所产⽣的全部责任及经济损失。（三）、" \
               "不得把所租车辆转借给其他⼈使⽤，不得将租赁车辆进⾏盈利性经营，以及参加竞赛，测验，实验，教练等活动。（四）、承担车辆租" \
               "赁期间的油料费⽤。在租赁期间应对⽔箱⽔位，制动液，冷却液负有每⽇检查的责任，在车辆正常使⽤中或者运⾏过程中，承租⽅应⽴" \
               "即通知出租⽅，并将车辆开⾄出租⽅指定的汽车修理⼚，承租⽅不得⾃⾏拆卸，更换车辆设备及装置，因⾮正常使⽤造成的事故责任" \
               "及损失费⽤均由承租⽅承担。（五）、按期如数缴纳租⾦，押⾦。（六）、应及时归还车辆，归还时的车况应与《租车车辆交接单》中的" \
               "车况登记⼀致，并经出租⽅指定的专业⼈员验收。验收时，发现车辆所有的划痕，刮伤，碰撞，损坏，设备折损，证件丢失等现象，承租" \
               "⽅应按实际损失缴纳车损费及其他相应的费⽤。（七）、如需续租车辆，需提前24⼩时到出租⽅办理⼿续。（⼋）、必须承担因承租⽅" \
               "的⾏为⽽带来的其他经济损失。 七、租车押⾦1、承租⽅在本合同签订时⼀次性向出租⽅交付⾜额押⾦，以《汽车租赁登记表》上所⽰" \
               "为准。2、在本合同期满或者双⽅协议解除合同时，如承租⽅⽆违约⾏为，出租⽅将押⾦归还给承租⽅。3、承租⼈在还车时需缴纳交通违" \
               "章保障⾦2000元⼈民币，在15个⼯作⽇后，经确定在租赁期间⽆违章记录后归还。 ⼋、车辆保险1、出租⽅为租赁车辆办理了车辆损失" \
               "险，盗抢险，⾃燃险，第三者责任险，承租⽅可⾃愿购买其他保险，费⽤由承租⽅⾃⾏承担。2、车辆在租赁期间若发⽣保险事故，承租⽅" \
               "应⽴即通知交通管理部门的出租⽅，出租⽅届时应协助承租⽅向保险公司报案，承租⽅必须协助出租⽅办理此事故的相关事宜，并⽀付" \
               "由此产⽣的⼀切费⽤。如属于保险赔付范围的费⽤由保险公司承担，属于保险责任免赔或其他原因导致保险公司拒赔的损失由承租⽅承担。" \
               " 九、违约责任（⼀）、出租⽅按合同规定时间提供给承租⽅车辆，每迟延⼀天，向承租⽅⽀付违约⾦100元。（⼆）、出租⽅所提供的" \
               "车辆的质量，型号必须按照承租合同的要求，如未按照承租合同的要求提供给承租⽅车辆，给承租⽅造成不便或损失，由出租⽅赔偿。" \
               "（三）、承租⽅要负责对所租车辆进⾏维护保养，在退租时如给车辆设备造成损坏，承租⽅应负责修复原状或赔偿，修复期照收租费。因出" \
               "租⽅所派司机驾驶不当造成损坏的由出租⽅⾃负，如果致使承租⽅不能按合同规定正常使⽤租赁车辆，承租⽅不但不给付出租⽅不能使⽤" \
               "期间的租费，⽽且出租⽅每天还要偿付承租⽅100元钱的违约⾦。（四）、出租⽅不得擅⾃将车调回，否则将按租⾦的双倍索赔承租⽅。承" \
               "租⽅必须按合同规定的时间和租⾦付款，否则，每逾期⼀天，加罚⼀天的租⾦。（五）、承租⽅⽆故辞退车辆，给出租⽅造成损失的，承租" \
               "⽅负责赔偿。因不可抗⼒，造成合同难以继续履⾏的，造成损失的，双⽅均不承担责任。 ⼗、合同的解除1、因车辆状况导致车辆⽆法正" \
               "常驾驶时，承租⽅有权解除合同。2、在未经出租⽅许可，承租⽅拖⽋3⽇以上租⾦，出租⽅有权随时随地解除合同。3、在下述任何⼀种情" \
               "况发⽣，出租⽅有权随时随地解除合同：⑴、承租⽅利⽤所租车辆进⾏违法犯罪活动。⑵承租⽅将所租车辆抵押，质押，转让，转租，出售" \
               "。⑶从事其他有损出租⽅车辆利益的情况 ⼗⼀、其他1、本合同履⾏期间，双⽅若发⽣争议，双⽅应协商解决，当事⼈不愿通过协商，调解" \
               "解决，或协商调解不成时，可以按照合同约定向浙江省杭州市仲裁委员会申请仲裁。2、其它未尽事项，由双⽅协商，另订附件。3、本" \
               "合同⼀式两份，双⽅各执正本⼀份，副本送有关管理机关备案。 甲⽅：天天租赁公司⼄⽅：悟空有限公司 2022年1⽉1⽇"
    m = MacbertCorrected()
    m(textlist)
