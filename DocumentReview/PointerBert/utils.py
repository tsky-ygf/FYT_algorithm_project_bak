#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/09/08 15:24
# @Author  : Czq
# @File    : utils.py
# @Software: PyCharm
import json
import os
import random
import pandas as pd
from pprint import pprint
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('model/language_model/chinese-roberta-wwm-ext')


def read_config(config_path):
    config_data = pd.read_csv(config_path, encoding='utf-8', na_values=' ', keep_default_na=False)
    config_list = []
    for line in config_data.values:
        config_list.append(line[0])
        # alis = line[1].split('|')
        # if alis:
        #     config_list.extend(alis)
    config_list = list(filter(None, config_list))
    return config_list

# 生成所有的通用label， 包含别称
def read_config_to_label(args):
    config_path = 'data/data_src/config.csv'
    # 读取config，将别称也读为schema
    config_list = read_config(config_path)

    # config_list.remove('争议解决')
    # config_list.remove('通知与送达')
    # config_list.remove('乙方解除合同')
    # config_list.remove('甲方解除合同')
    # config_list.remove('未尽事宜')
    # config_list.remove('附件')
    # TODO: 之前有保留金额, 在生成origin.json时
    # config_list.remove('金额')
    return config_list

# 加载train和dev数据
def load_data(path):
    out_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line.strip())
            # out_data.append([json_line['content'],json_line['result_list'],json_line['prompt']])
            out_data.append(json_line)
    return out_data


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


class ReaderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def batchify(batch):
    sentences = []
    labels = []
    input_ids = []
    attention_mask = []
    token_type_ids = []
    window_length = 510  # add 101 102 to 512
    start_seqs = []
    end_seqs = []

    for b in batch:
        content_raw = b['content']
        label = b['prompt']
        res_list = b['result_list']
        # negative ratio 生成的负例
        if not res_list:
            content = content_raw[:window_length]
            start_seq = [[0] * window_length for _ in range(len(labels2id))]
            end_seq = [[0] * window_length for _ in range(len(labels2id))]
            start_seqs.append(start_seq)
            end_seqs.append(end_seq)
            labels.append(label)
            sentences.append(content)
        else:
            # 如果 result_list 数量大于1：
            #       1. 如果多个result_list能在一个window中， 实体的最外边界往左右扩充至window
            #       2. 如果多个result_list不能在一个window中， 则视作多条样本
            # TODO: if length of res_list are more than one
            # assert len(res_list) == 1, ['length of result_list is larger than 1', b]
            starts_tmp = []
            ends_tmp = []
            for i_res in range(len(res_list)):
                start, end = res_list[i_res]['start'], res_list[i_res]['end']
                starts_tmp.append(start)
                ends_tmp.append(end)
            start_range = min(starts_tmp + ends_tmp)
            end_range = max(starts_tmp + ends_tmp)
            diff = end_range - start_range
            # 如果最大范围小于window大小
            if diff< 510:
                window_length_single = (window_length - diff) // 2
                left = start_range - window_length_single if start_range - window_length_single >= 0 else 0
                right = end_range + window_length_single if end_range + window_length_single < len(content_raw) else len(content_raw)
                content = content_raw[left:right]
                start_seq = [[0] * window_length for _ in range(len(labels2id))]
                end_seq = [[0] * window_length for _ in range(len(labels2id))]
                for i_res in range(len(res_list)):
                    start, end = res_list[i_res]['start'], res_list[i_res]['end']
                    start_new = start - left
                    end_new = end - left

                    label_index = labels2id.index(label)
                    start_seq[label_index][start_new] = 1
                    end_seq[label_index][end_new] = 1
                start_seqs.append(start_seq)
                end_seqs.append(end_seq)
                labels.append(label)
                sentences.append(content)

            else:
                for i_res in range(len(res_list)):
                    start, end = res_list[i_res]['start'], res_list[i_res]['end']
                    diff = end - start
                    window_length_single = (window_length - diff) // 2
                    left = start - window_length_single if start - window_length_single >= 0 else 0
                    right = end + window_length_single if end + window_length_single < len(content_raw) else len(content_raw)
                    start = start - left
                    end = end - left
                    content = content_raw[left:right]
                    assert content[start:end] == res_list[i_res]['text']
                    # 所有的标签都需要start和end
                    start_seq = [[0] * window_length for _ in range(len(labels2id))]
                    label_index = labels2id.index(label)
                    start_seq[label_index][start] = 1
                    end_seq = [[0] * window_length for _ in range(len(labels2id))]
                    end_seq[label_index][end] = 1
                    start_seqs.append(start_seq)
                    end_seqs.append(end_seq)
                    labels.append(label)
                    sentences.append(content)

        # enco_dict = tokenizer.encode_plus(list(content), padding='max_length', add_special_tokens=True, max_length=510,
        #                                   return_tensors='pt', truncation=True)
        # input_id = enco_dict['input_ids'].squeeze()
        # atten_mask = enco_dict['attention_mask'].squeeze()
        # token_type_id = enco_dict['token_type_ids'].squeeze()
        # if len(input_id) != len(content)+2:
    for senten in sentences:

        input_i = [101] + tokenizer.convert_tokens_to_ids(list(senten)) + [102]
        input_id = input_i.copy() + [0] * (512 - len(input_i))
        atten_mask = [1] * len(input_id) + [0] * (512 - len(input_id))
        token_type_id = [0] * 512

        assert len(input_id) == 512, len(input_id)
        input_ids.append(input_id)
        attention_mask.append(atten_mask)
        token_type_ids.append(token_type_id)

    encoded_dict = {
        'input_ids': torch.LongTensor(input_ids).to('cuda'),
        'attention_mask': torch.LongTensor(attention_mask).to('cuda'),
        'token_type_ids': torch.LongTensor(token_type_ids).to('cuda')
    }
    start_seqs = torch.FloatTensor(start_seqs).transpose(1, 2).to('cuda')
    end_seqs = torch.FloatTensor(end_seqs).transpose(1, 2).to('cuda')
    assert len(input_ids) == len(start_seqs), [len(input_ids), len(start_seqs)]
    return encoded_dict, start_seqs, end_seqs, labels, sentences


if __name__ == "__main__":
    # TODO guolv le ？
    # j = {"content": "母婴护理（月嫂）服务合同甲方（母婴护理公司）：诺依曼软件传媒有限公司服务电话：13078797720经营地址：北京市淑珍市丰都济南路E座277335号乙方（月嫂）：姜证和身份证号：420101198108146156联系电话：13754621600原籍地址：海南省巢湖市房山阴路T座391402号根据《中华人民共和国合同法》及有关法律、法规的规定，甲、乙双方本着平等、自愿、诚实信用的原则，就母婴护理服务相关事宜签订本合同。第一条甲、乙双方的关系甲方和乙方是劳务合同关系，甲方委派乙方为用户服务。第二条服务内容和服务地点甲方推荐并委派符合上岗条件（体检合格，经过岗前培训并取得行业或政府相关部门认可的培训合格证书）的乙方到服务地点为用户承担照料孕、产妇与新生儿及其他服务。第三条服务期限1、服务期限180天，从2022年1月5日起至2022年7月4日止，实际服务时间从乙方到岗之日起算。2、服务时间每月按30天计，保证服务工作日26天，休息4天，其中遇国家节假期10日，服务报酬为日工资的三倍，即300元/天。第四条服务报酬的支付期限与方式1、乙方的服务报酬为3000元/月。凡不足月者按26天/月的日平均服务报酬结算。2、乙方完成一个月工作期满之日起5天内到甲方处领取服务报酬并签字确认。3、乙方因被用户投诉、中途退回或乙方自身原因提前要求离岗，乙方的服务报酬根据服务合同约定标准按实际工作天数的80%计发。第五条甲方的义务和权利1、为乙方建立个人资料和服务档案，全面记录乙方的工作经历和评价，对乙方有效的身份证、居住证、健康体检报告复印存档。2、向乙方提供用户的真实详细地址、姓名等资料，以及用户母婴和家庭成员中有无精神病、传染病和其它重大疾病情况。3、负责乙方的岗前培训、教育和后续的管理工作，加强对乙方的思想教育、技能培训和监督指导。积极引导乙方按《母婴护理工作范围》（附件一）和《婴儿安全问题及意外防护措施》（附件二）进行操作。4、建立母婴护理服务质量管理制度，定期了解乙方的服务情况，指导督促乙方执行合同的各项约定，并对乙方的不良行为进行批评指正。5、确保用户在服务质量上的利益和维护乙方的合法权益。接受用户与乙方的投诉并进行核实，协调用户与乙方的关系，妥善处理投诉和调换要求，协助解决乙方、用户双方产生的纠纷。因用户的原因造成乙方损失的，甲方应出面协调解决，并先行承担相关责任。6、为乙方购买《家政人员意外伤害保险》。7、乙方未经甲方和用户的同意擅自离岗，甲方有权不支付服务报酬。第六条乙方的义务和权利1、上岗前应如实填写本人身份及家庭情况等基本资料，提供有效真实的身份证、健康证和培训证等资料。2、遵守国家的法律法规，不得损害用户的合法权益，如因工作失误而造成用户人身或其他权益受侵害的，则要承担相应的法律责任和经济赔偿责任。3、善待服务对象，禁止擅自离岗。与用户发生纠纷应及时向甲方反映，经甲方同意方可中止提供服务。如确需提前终止合同，应提前5天提出，并到甲方处办理手续后，方能离开用户家。4、服从甲方和用户的管理和指导，尊重用户的生活习俗，工作认真负责，勤俭节约，经手的钱物帐目清楚。注意个人卫生，禁止佩戴首饰、禁用化妆品。工作时间不得接听电话、看电视和做私事，如遇特殊情况必须向用户说明。5、遵守职业道德，不得在用户住处从事与服务范围无关的活动；不得擅自将他人及亲友带入或住宿用户家中；不得擅自翻动、拿用用户的物品，不得向用户索要财物，更不得有偷窃及破坏行为；不得擅自动用高档电器和贵重物品；不参与用户家庭内部事务和邻里纠纷；不泄露和传播用户的家庭隐私和个人信息；爱护用户家庭和财产；不得擅自给婴儿喂食药物及有害食物，不得推荐药品或保健品给婴儿及产妇；不得违反《母婴护理工作范围》及《婴儿安全问题及意外防护措施》进行操作。6、请假外出应征得用户同意。采取实时照休的休假办法的要按日平均服务报酬数扣除相应的服务报酬。采取集中补休的休假办法的可作调休而不扣除相应的服务报酬。请假外出应告知去向，不得在外留宿，如遇特殊情况不能按时返回的，应提前通知用户。7、按时得到服务报酬，以及得到正常的休假和休息时间。若用户占用乙方休假或休息时间，乙方有权要求支付加班服务报酬。8、有权保护自己人身和名誉不受侵犯，追究因甲方或用户过错造成的经济损失及法律责任，并与甲方解除合同。9、有权拒绝从事与合同内容不符的工作，有权拒绝为第三方服务，有权拒绝在非约定地址服务。第七条甲、乙双方的特别约定1、乙方离岗时，甲方应主动提醒用户认真检查家庭财物有无损坏和丢失。乙方离岗后，不再承担用户财物损失责任。2、乙方私自与用户续约，因此而产生的法律责任由乙方与用户承担。3、乙方与用户本合同期满后3个月内不能建立雇用关系，否则应向甲方支付赔偿金10000元。第八条违约责任1、甲方逾期支付乙方服务报酬的，每逾期1天按应付服务报酬1%向乙方支付逾期付款的违约金。2、乙方不按甲方的指示为用户提供服务而产生的一切损失由乙方承担，包括因此而致使甲方向用户承担违约责任的违约金。第九条合同争议的解决办法本合同如果发生争议，双方协商解决，协商不成的，可向广州家庭服务业协会或广州市消费者委员会申请调解。协商或调解不成的，按下列第1种方式解决：1、提交广州仲裁委员会仲裁。2、向人民法院起诉。第十条合同未尽事宜及生效本合同未尽事宜双方另行协商补充。补充协议与本合同具有同等法律效力。本合同一式二份，甲、乙双方各执一份，具有同等法律效力，双方签字或盖章之日起生效。甲方（盖章）：诺依曼软件传媒有限公司法定代表人（签字）：芮凤英2022年1月5日乙方（签字）：姜证和2022年1月5日", "result_list": [{"text": "3000元/月", "start": 480, "end": 487}], "prompt": "金额"}
    # t = j['content']
    # start = t.index('3000元/月')
    # end = start+len('3000元/月')
    # print(start, end)
    #
    # assert False
    file = "ContractNER/data_src/new/train.txt"
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            index = []
            if len(json_line['result_list']) > 1:
                rs = json_line['result_list']
                for r in rs:
                    index.append(r['start'])
                    index.append(r['end'])
                if max(index) - min(index) > 510:
                    assert False, [rs, json_line['prompt']]
    pass
else:
    labels2id = read_config_to_label(None)
