# -*- coding: utf-8 -*-
import sys
import os
import re
import json
import pymysql
import pandas as pd
import random
import jieba
import jieba.posseg
import numpy as np
import traceback
import multiprocessing
from pyltp import Parser, Segmentor, Postagger

"""
util文件。主要功能包括：提取标签(y值）；得到特征列表
"""
fact_reason_splitter = "\t"

new_problems = ['劳动纠纷', '社保纠纷', '工伤赔偿', '劳务纠纷', '提供劳务者受害责任纠纷', '提供劳务者致害责任纠纷']
table_status = 2

host_ip = "rm-bp18g0150979o8v4t.mysql.rds.aliyuncs.com"
# host_ip = "rm-bp18g0150979o8v4t.mysql.rds.aliyuncs.com"

# 否定词，分词后否定词列表，不在列表里的没有否定意义
negative_word_list = ['不', '未', '没', '没有', '无', '非', '并未', '不再', '不能', '无法', '不足以', '不存在', '不能证明', '不认可','尚未']
negative_word = '(' + '|'.join(negative_word_list) + ')'  # 解除

database = {
    '工伤赔偿': 'case_list_original_labor_2',
    '社保纠纷': 'case_list_original_labor_2',
    '劳动纠纷': 'case_list_original_labor',
    '婚姻家庭': 'case_list_original_hunyinjiating',
    '继承问题': 'case_list_original_hunyinjiating',
    # '物业纠纷': 'case_list_original_hetong',
    '劳务纠纷': 'case_list_original_hetong',
    '交通事故': 'case_list_original_qinquan',
    '租赁纠纷': 'case_list_original_hetong',
    '买卖纠纷': 'case_list_original_hetong',
    '借贷纠纷': 'case_list_original_hetong',
    '房产纠纷': 'case_list_original_hetong',
    # '银行卡纠纷': 'case_list_original_hetong',
    '借记卡纠纷': 'case_list_original_hetong',
    '保管合同纠纷': 'case_list_original_hetong',
    '保证合同纠纷': 'case_list_original_hetong',
    '承揽合同纠纷': 'case_list_original_hetong',
    '缔约过失责任纠纷': 'case_list_original_hetong',
    '定金合同纠纷': 'case_list_original_hetong',
    '服务合同纠纷': 'case_list_original_hetong',
    '广告合同纠纷': 'case_list_original_hetong',
    '建设工程分包合同纠纷': 'case_list_original_hetong',
    '建设工程设计合同纠纷': 'case_list_original_hetong',
    '借用合同纠纷': 'case_list_original_hetong',
    '金融借款合同纠纷': 'case_list_original_hetong',
    '居间合同纠纷': 'case_list_original_hetong',
    '施工合同纠纷': 'case_list_original_hetong',
    '农林牧渔承包合同纠纷': 'case_list_original_hetong',
    # '企业借贷纠纷': 'case_list_original_hetong',
    '物业服务合同纠纷': 'case_list_original_hetong',
    '运输合同纠纷': 'case_list_original_hetong',
    '赠与合同纠纷': 'case_list_original_hetong',
    '装饰装修合同纠纷': 'case_list_original_hetong',
    '农村土地承包合同纠纷': 'case_list_original_hetong',
    '建设用地使用权合同纠纷': 'case_list_original_hetong',
    '信用卡纠纷': 'case_list_original_hetong',
    '仓储合同纠纷': 'case_list_original_hetong',
    '典当纠纷': 'case_list_original_hetong',
    '房屋拆迁安置补偿合同纠纷': 'case_list_original_hetong',
    '合伙协议纠纷': 'case_list_original_hetong',
    '委托合同纠纷': 'case_list_original_hetong',
    '民间委托理财合同纠纷': 'case_list_original_hetong',
    '建设用地使用权纠纷': 'case_list_original_wuquan',
    '农村土地承包经营权纠纷': 'case_list_original_wuquan',
    '占有保护纠纷': 'case_list_original_wuquan',
    '宅基地使用权纠纷': 'case_list_original_wuquan',
    '相邻关系纠纷': 'case_list_original_wuquan',
    '物权保护纠纷': 'case_list_original_wuquan',
    '侵害集体组织成员权益纠纷': 'case_list_original_wuquan',
    '共有纠纷': 'case_list_original_wuquan',
    '抵押权纠纷': 'case_list_original_wuquan',
    '承包地征收补偿费用分配纠纷': 'case_list_original_wuquan',
    '产品责任纠纷': 'case_list_original_qinquan',
    '触电人身损害责任纠纷': 'case_list_original_qinquan',
    '地面施工、地下设施损害责任纠纷': 'case_list_original_qinquan',
    '公共场所管理人责任纠纷': 'case_list_original_qinquan',
    '公共道路妨碍通行损害责任纠纷': 'case_list_original_qinquan',
    '监护人责任纠纷': 'case_list_original_qinquan',
    '建筑物、构筑物倒塌损害责任纠纷': 'case_list_original_qinquan',
    '教育机构责任纠纷': 'case_list_original_qinquan',
    '饲养动物致人损害责任纠纷': 'case_list_original_qinquan',
    '提供劳务者受害责任纠纷': 'case_list_original_qinquan',
    '提供劳务者致害责任纠纷': 'case_list_original_qinquan',
    '铁路运输人身损害责任纠纷': 'case_list_original_qinquan',
    '网络侵权责任纠纷': 'case_list_original_qinquan',
    '物件脱落、坠落损害责任纠纷': 'case_list_original_qinquan',
    '医疗损害责任纠纷': 'case_list_original_qinquan',
    '义务帮工人受害责任纠纷': 'case_list_original_qinquan',
    '用人单位责任纠纷': 'case_list_original_qinquan',
    '隐私权纠纷': 'case_list_original_rengequan',
    '一般人格权纠纷': 'case_list_original_rengequan',
    '姓名权纠纷': 'case_list_original_rengequan',
    '肖像权纠纷': 'case_list_original_rengequan',
    '生命权、健康权、身体权纠纷': 'case_list_original_rengequan',
    '名誉权纠纷': 'case_list_original_rengequan',
}

online_problems2 = ['劳动纠纷', '社保纠纷', '婚姻家庭', '工伤赔偿', '继承问题', '交通事故',
                    '借记卡纠纷', '房产纠纷', '租赁纠纷', '买卖纠纷', '借贷纠纷', '劳务纠纷']
online_problems31 = ['服务合同纠纷', '运输合同纠纷']
online_problems32 = ['金融借款合同纠纷', '居间合同纠纷', '农林牧渔承包合同纠纷', '保管合同纠纷', '定金合同纠纷',
                     '物业服务合同纠纷', '信用卡纠纷', '赠与合同纠纷', '装饰装修合同纠纷', '借用合同纠纷',
                     '建设工程设计合同纠纷', '施工合同纠纷', '缔约过失责任纠纷', '仓储合同纠纷',
                     '保证合同纠纷', '典当纠纷', '农村土地承包合同纠纷', '合伙协议纠纷', '委托合同纠纷',
                     '广告合同纠纷', '建设用地使用权合同纠纷', '房屋拆迁安置补偿合同纠纷', '承揽合同纠纷',
                     '民间委托理财合同纠纷', '建设工程分包合同纠纷']
online_problems33 = ['隐私权纠纷', '一般人格权纠纷', '姓名权纠纷', '肖像权纠纷', '生命权、健康权、身体权纠纷', '名誉权纠纷',
                     '产品责任纠纷', '触电人身损害责任纠纷', '地面施工、地下设施损害责任纠纷', '公共场所管理人责任纠纷',
                     '公共道路妨碍通行损害责任纠纷', '监护人责任纠纷', '建筑物、构筑物倒塌损害责任纠纷',
                     '教育机构责任纠纷', '饲养动物致人损害责任纠纷', '提供劳务者受害责任纠纷', '提供劳务者致害责任纠纷',
                     '铁路运输人身损害责任纠纷', '网络侵权责任纠纷', '物件脱落、坠落损害责任纠纷', '医疗损害责任纠纷',
                     '义务帮工人受害责任纠纷', '用人单位责任纠纷', '建设用地使用权纠纷', '农村土地承包经营权纠纷',
                     '占有保护纠纷', '宅基地使用权纠纷', '相邻关系纠纷', '物权保护纠纷', '侵害集体组织成员权益纠纷',
                     '共有纠纷', '抵押权纠纷', '承包地征收补偿费用分配纠纷']
online_problems = list(database.keys())

problem_desc_dict = {}
problem_desc_dict[
    '婚姻家庭'] = '男女双方自愿/不自愿登记结婚，婚后育有x子/女，现x岁，因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况。'
problem_desc_dict[
    '交通事故'] = '驾驶车辆撞到其他车辆或行人造成交通事故的具体经过，驾驶者是否为车主本人，车上是否还有其他乘客及是否受伤、是否取得《交通事故认定书》，认定书的具体内容及责任划分，是否为肇事车辆投保及保险类型。'
problem_desc_dict['借贷纠纷'] = '双方签订《借款协议》或有借条、欠条、收据等，已通过转账/现金方式支付款项，用于xx用途，借款人到期未足额偿还本金/利息/逾期利息。'
problem_desc_dict['劳动纠纷'] = '某年某月某日，到某处工作，双方签订/未签订劳动合同，约定工资x元。工作期间，因xx原因导致双方于某年某月某日解除劳动合同或遭用人单位辞退。'
problem_desc_dict['房产纠纷'] = '双方签订《房屋买卖合同》/其他协议，约定的内容，双方违约的具体情况。'
problem_desc_dict['买卖纠纷'] = '双方签订《xx买卖合同》/其他协议，约定的内容，双方违约的具体情况。'
problem_desc_dict['租赁纠纷'] = '双方签订《租房合同》/其他协议，约定的内容，双方违约的具体情况。'
problem_desc_dict['工伤赔偿'] = '某年某月某日，到某处工作，双方签订/未签订劳动合同，约定工资x元，工作期间，因xx原因受工伤，是否已申请工伤认定/伤残鉴定，以及认定/鉴定结果。'
problem_desc_dict['劳务纠纷'] = '某年某月某日，到某处工作，双方签订/未签订劳务协议，约定工资x元，已拖欠工资x元。'
problem_desc_dict['社保纠纷'] = '某年某月某日，到某处工作，双方签订/未签订劳动合同，约定工资x元，公司某年某月某日开始办理/未办理社会保险，是否存在未足额缴纳保险费及非因本人意愿中断的情形。'
problem_desc_dict['继承问题'] = '某某死后留有xx遗产（是否有遗嘱/扶养协议），某某拥有继承权，继承人尽/未尽到赡养义务的具体情形。'
problem_desc_dict['借记卡纠纷'] = '向某银行账户存储x元，发现账户内存款被盗刷或减少，银行存在xx过错，本人是否报警或挂失。'
problem_desc_dict['保管合同纠纷'] = 'x年x月x日，x将物品交给x保管，签订/未签订保管合同，约定/未约定保管费，保管人未尽保管义务导致保管物毁损或未按时归还、委托人拖欠保管费等。'
problem_desc_dict['保证合同纠纷'] = 'x年x月x日，x向x提供保证，签订/未签订保证合同，约定保证范围为x，保证方式为x，保证期限届满，x违反约定存在x行为，造成x后果。'
problem_desc_dict[
    '承揽合同纠纷'] = 'x年x月x日，x委托x加工定做x，双方签订/未签订承揽加工合同，约定费用x元，一方履行/未履行合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在没有达到维修标准、存在质量问题等行为，造成x后果。'
problem_desc_dict['定金合同纠纷'] = 'x年x月x日，为x支付了定金x元，x违反约定导致合同目的无法实现。'
problem_desc_dict['借用合同纠纷'] = 'x年x月x日，x将物品借给x使用，双方签订/未签订借用合同，约定/未约定借用费，x违反约定导致借用物毁损或借用人未按时归还借用物/拖欠借用费等。'
problem_desc_dict[
    '金融借款合同纠纷'] = 'x年x月x日，双方签订/未签订金融借款合同纠纷，借款的用途为x，出借人交付/未交付款项，借款利率利息为x%，逾期利息为x%，双方存在x违约行为如未按约定还款、未按约偿还借款本息等。'
problem_desc_dict[
    '居间合同纠纷'] = 'x年x月x日，双方签订/未签订居间合同，约定x向x提供x居间服务，居间方履行/未履行x居间义务导致x委托事项完成/未完成，委托方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果。'
problem_desc_dict[
    '农林牧渔承包合同纠纷'] = 'x年x月x日，x承包了x，双方签订/未签订承包合同协议，承包方支付/未支付承包费，发包方交付/未交付约定的土地/林地/池塘，及合同履行过程中x违反约定存在x行为，造成x后果。'
problem_desc_dict['委托合同纠纷'] = 'x年x月x日，x委托x做x事，双方签订/未签订委托合同，一方履行/未履行合同，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果。'
problem_desc_dict['物业服务合同纠纷'] = 'x年x月x日，双方签订/未签订物业服务合同，物业公司提供履行/未提供物业服务，业主支付/未支付物业费，及合同履行过程中x违反约定存在x行为，造成x后果。'
problem_desc_dict['信用卡纠纷'] = 'x年x月x日，x向x银行申请办理信用卡，银行发放/未发放信用卡，x使用/未使用该信用卡消费，或透支/未透支使用，如期/未如期偿还，或存在其他行为造成x后果。'
problem_desc_dict['赠与合同纠纷'] = 'x年x月x日，签订/口头约定赠与合同，赠与物为x，赠与人交付/未交付赠与物，x违反x义务。'
problem_desc_dict['装饰装修合同纠纷'] = 'x年x月x日，签订/未签订装修装饰合同，装修方提供/未提供装修服务，另一方支付/未支付装修款，及其他违约行为如出现质量问题等。'
problem_desc_dict['仓储合同纠纷'] = 'x年x月x日，签订/未签订仓储合同，一方当事人提供/未提供仓储服务，另一方支付/未支付费用，x存在x违约行为如贬值、损坏、擅自移动、发生火灾等。'
problem_desc_dict['缔约过失责任纠纷'] = 'x年x月x日，计划签订x合同，在订立合同的过程中一方违背了诚实信用原则作出x行为如恶意进行磋商、不具备公开出售条件、提供虚假情况等，合同成立/未成立，给另一方造成了x损失。'
problem_desc_dict['典当纠纷'] = 'x年x月x日，签订/未签订典当合同，一方支付/未支付当金，典当期满，x逾期赎当/续当，支付/未偿还当金本息、综合费，x存在x违约行为如没有赎当、无力还款等。'
problem_desc_dict['房屋拆迁安置补偿合同纠纷'] = 'x年x月x日，签订/未签订房屋拆迁安置补偿合同，约定如过渡费、补偿款、安置房等内容，x存在x违约行为如房屋未交付、无法入住等。'
problem_desc_dict['广告合同纠纷'] = 'x年x月x日，签订/未签订广告合同，一方提供/未提供广告服务，另一方支付/未支付广告费，x存在x违约行为如拒绝付款等。'
problem_desc_dict['合伙协议纠纷'] = 'x年x月x日，签订/未签订合伙协议，x全面履行出资义务/存在x违约行为，合伙期间债权债务情况。 '
problem_desc_dict[
    '建设工程分包合同纠纷'] = 'x年x月x日，双方签订/未签订建设工程分包合同，一方履行/未履行合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为如违法分包、没有资质等，造成x后果。'
problem_desc_dict['建设工程设计合同纠纷'] = 'x年x月x日，双方签订/未签订建设工程设计合同，一方履行/未履行合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果。'
problem_desc_dict['建设用地使用权合同纠纷'] = 'x年x月x日，签订/未签订建设用地使用权合同，一方履行/未履行合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果。'
problem_desc_dict[
    '民间委托理财合同纠纷'] = 'x年x月x日，签订/未签订民间委托理财合同，x全面履行合同义务/存在x违约行为，在委托期限届满时，足额/未足额返还投资款、约定的收益，及其他违约行为如拒绝付款、拒不归还、占为己有等。'
problem_desc_dict['农村土地承包合同纠纷'] = 'x年x月x日，x承包x土地，双方签订/未签订土地承包合同协议，承包方支付/未支付承包费，发包方交付/未交付约定的土地，及合同履行过程中x违反约定存在x行为，造成x后果。'
problem_desc_dict['施工合同纠纷'] = 'x年x月x日，签订/未签订建设施工合同，合同履行过程中x违反约定存在x行为如拖欠施工款、未开工、质量缺陷等，给另一方造成x损失。'
problem_desc_dict[
    '服务合同纠纷'] = 'x年x月x日，x在x处办理x业务/会员卡/到某处消费等，x提供/未提供服务，x支付/未支付服务费，或x存在x违约行为如私自切断信号、根本无法正常使用、额外扣费等，造成x损害后果，如致使受伤、旅客财物被盗等。'
problem_desc_dict['运输合同纠纷'] = 'x年x月x日，x与x签订运输合同/搭乘车辆，x履行/未履行运输义务，x支付/未支付货款，或x存在x违约行为，造成x损害后果，如突然刹车，与客车相撞等。'
problem_desc_dict['名誉权纠纷'] = 'x年x月x日，x实施了x侵权行为，如诽谤、散布诽谤言论等，侵权人是否故意，对x造成了x损害后果，如导致精神负担、造成经济损失等。'
problem_desc_dict['肖像权纠纷'] = 'x年x月x日，x实施了x侵权行为，如用照片进行商业宣传，侵权人是否故意，对x造成了x损害后果，如给精神造成损害等'
problem_desc_dict['姓名权纠纷'] = 'x年x月x日，x实施了x侵权行为，如冒名担保、盗用身份证复印件等，侵权人是否故意，对x造成了x损害后果，如经济受损害、造成精神伤害等。'
problem_desc_dict['一般人格权纠纷'] = 'x年x月x日，x实施了x侵权行为，如辱骂、骚扰等，侵权人是否故意，对x造成了x损害后果，如经济严重损失、伤害身心健康等。'
problem_desc_dict['隐私权纠纷'] = 'x年x月x日，x实施了x侵权行为，如泄露个人信息、窃取个人信息等，侵权人是否故意，对x造成了x损害后果，如引发不良心理状态、造成精神伤害等。'
problem_desc_dict['生命权、健康权、身体权纠纷'] = 'x年x月x日，x实施了x侵权行为，如砸伤他人等，侵权人是否故意，对x造成了x损害后果，如蒙受经济损失、导致死亡等。'
problem_desc_dict['产品责任纠纷'] = 'x年x月x日，购买了x产品，产品存在x问题，如不符合国家安全标准、质量存在问题等，造成了x损害后果，如财物毁损、造成严重烧伤等。'
problem_desc_dict['提供劳务者致害责任纠纷'] = 'x年x月x日，签订/未签订劳务合同/协议，劳务人员在提供劳务期间存在x行为，如砸到他人等，造成x后果，如骨折、致多人伤亡等。'
problem_desc_dict[
    '铁路运输人身损害责任纠纷'] = 'x年x月x日，签订运输合同/搭乘车辆，期间存在x违约行为，如将人摔伤、将人撞倒等，侵权人是否故意或重大过失，如未设警示、管理不善，造成x损害后果，如骨折、高位截肢等。'
problem_desc_dict[
    '提供劳务者受害责任纠纷'] = 'x年x月x日，签订/未签订劳务合同/协议，在提供劳务期间x存在x行为，如被砸伤等，导致劳务人员造成x损害后果，如导致骨折、多处伤残等，侵权人是否故意或重大过失，如未严格制定操作规程、未尽到监管义务等。'
problem_desc_dict['用人单位责任纠纷'] = 'x年x月x日，签订/未签订劳动合同，劳动者在执行职务/非执行职务的过程中，存在x行为，如砸伤第三人等，对受害人造成x损害后果，如导致第三人死亡、骨折等。'
problem_desc_dict[
    '教育机构责任纠纷'] = 'x年x月x日，受害人在x学校接受教育，在校期间，遭遇了x行为，如被打伤、被猥亵等，侵权人是否故意或重大过失如看护不力、未尽管理职责等，对受害人造成了x损害后果，如心理障碍、造成身心受到伤害等。'
problem_desc_dict['监护人责任纠纷'] = 'x年x月x日，x对受害人实施了x行为，如戳伤、撞倒等，造成了x损害后果，如骨折、全身多处擦伤等。'
problem_desc_dict['公共场所管理人责任纠纷'] = 'x年x月x日，在餐馆、宾馆等公共场所），遭受了x损害，如遭到殴打、被捅死等，管理人存在x过错，如安全警示标志设立不规范、未尽监督管理责任等。'
problem_desc_dict['义务帮工人受害责任纠纷'] = 'x年x月x日，x为x提供义务帮工，被帮工人明确/未明确拒绝，帮工人收取/未收取报酬，因x原因导致帮工人受到x伤害，如致使物品坠落、被物品砸伤等。'
problem_desc_dict['医疗损害责任纠纷'] = 'x年x月x日，x在医院接受治疗，因医院x行为，如诊治失误、销毁病历等，造成x损害后果，如留下后遗症、病情加重等。'
problem_desc_dict['物件脱落、坠落损害责任纠纷'] = 'x年x月x日，x在x场所，因物件脱落/坠落造成x损害后果，如高空掉下某物被砸伤等，管理人是否尽到管理义务。'
problem_desc_dict['建筑物、构筑物倒塌损害责任纠纷'] = 'x年x月x日，x在x场所，因建筑物/构筑物倒塌，如断裂、坍塌、被砸伤等，对受害人造成x损害后果，如生活不能自理、当场死亡等，管理人是否尽到管理义务。'
problem_desc_dict['公共道路妨碍通行损害责任纠纷'] = 'x年x月x日，x在x道路，因公共道路妨碍通行，如在路面堆放物品导致车辆侧翻等，造成x损害结果，如致伤残、脑震荡等，道路管理者是否尽到管理提醒义务。'
problem_desc_dict['地面施工、地下设施损害责任纠纷'] = 'x年x月x日，x在x道路，因地面施工/地下设施，造成x损害结果，如致使翻车、导致爆胎等，道路管理者是否尽到管理提醒义务。'
problem_desc_dict['触电人身损害责任纠纷'] = 'x年x月x日，x因触电造成x损害结果，如被电晕、被电伤等。'
problem_desc_dict['饲养动物致人损害责任纠纷'] = 'x年x月x日，x被x动物伤害，如被撞倒、猛咬等，造成x损害后果，如狂犬病、骨折等，动物饲养人是否尽到饲养义务，受害者是否存在故意挑逗动物行为。'
problem_desc_dict['网络侵权责任纠纷'] = 'x年x月x日，x通过网络进行了x行为，如发布贬损言论、散布负面帖文等，对x造成了x损害后果，如造成商誉降低等。'
problem_desc_dict['宅基地使用权纠纷'] = 'x年x月x日，是/不是该村村民，通过x方式取得该宅基地使用权，在此过程中x存在x行为影响到该宅基地的使用。'
problem_desc_dict['占有保护纠纷'] = 'x是/不是该物的所有人或通过x方式对该物享有x权利，x年x月x日，该物被x侵权人非法占有拒不归还或受到x损害导致了x后果。'
problem_desc_dict['承包地征收补偿费用分配纠纷'] = 'x年x月x日，x承包地被征收，补偿款为x元，通过x种方式取得该承包地经营权，承包地被征收时，由于x原因未足额分得应得补偿款或存在x争议。'
problem_desc_dict['抵押权纠纷'] = 'x年x月x日，为何事签订/未签订抵押合同，办理/未办理抵押登记，抵押期限x个月/年，抵押期限届满，债务清偿/未清偿完毕或满足/不满足抵押权实现条件。'
problem_desc_dict['物权保护纠纷'] = 'x是/不是该物的所有人或通过x方式对该物享有x权利，x年x月x日，该物被x侵权人非法占有拒不归还或受到x损害导致了x后果。'
problem_desc_dict['侵害集体组织成员权益纠纷'] = 'x是/不是本村民，在本村给村民分配权益时，存在/不存在不公平分配情形，造成x损害后果。'
problem_desc_dict['相邻关系纠纷'] = 'x与x存在/不存在相邻关系，x存在x行为，如擅自改建、排放粪便等，给另一方当事人造成x损害后果，如影响排水、对日照构成影响等。'
problem_desc_dict['共有纠纷'] = 'x物属于/不属于x与x的共有物，存在/不存在协议，现因x原因请求确认享有共有权/分割共有物。'
problem_desc_dict['农村土地承包经营权纠纷'] = 'x年x月x日，通过x种方式承包x村土地/林地，取得/未取得该承包地经营权，现该地存在x情形，导致该地无法正常使用。'
problem_desc_dict[
    '建设用地使用权纠纷'] = 'x年x月x日，通过划拨、出让或转让取得x建设用地使用权，签订/未签订建设用地使用权合同，办理/未办理相关手续，由于x行为导致x后果，被侵犯x合法权益造成该建设用地及其建筑物无法正常使用。'

# problem_desc_dict['婚姻家庭'] = '男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）'
# problem_desc_dict['交通事故'] = '驾驶车辆撞到其他车辆或行人造成交通事故的具体经过（驾驶者是否为车主本人，车上是否还有其他乘客及是否受伤、是否取得《交通事故认定书》，认定书的具体内容及责任划分，是否为肇事车辆投保及保险类型）。'
# problem_desc_dict['借贷纠纷'] = '双方签订《借款协议》或有借条、欠条、收据等，已通过转账/现金方式支付款项，用于xx用途，借款人到期未足额偿还本金/利息/逾期利息。（借款期限、利息年利率x、逾期利息年利率x）。'
# problem_desc_dict['劳动纠纷'] = '某年某月某日，到某处工作，双方签订/未签订劳动合同，约定工资x元。工作期间，因xx原因导致双方于某年某月某日解除劳动合同或遭用人单位辞退。'
# problem_desc_dict['房产纠纷'] = '双方签订《房屋买卖合同》/其他协议，约定的内容，双方违约的具体情况。'
# problem_desc_dict['买卖纠纷'] = '双方签订《xx买卖合同》/其他协议，约定的内容，双方违约的具体情况。'
# problem_desc_dict['租赁纠纷'] = '双方签订《租房合同》/其他协议，约定的内容，双方违约的具体情况。'
# problem_desc_dict['工伤赔偿'] = '某年某月某日，到某处工作，双方签订/未签订劳动合同，约定工资x元，工作期间，因xx原因受工伤（是否已申请工伤认定/伤残鉴定，以及认定/鉴定结果）。'
# problem_desc_dict['劳务纠纷'] = '某年某月某日，到某处工作，双方签订/未签订劳务协议，约定工资x元，已拖欠工资x元。'
# problem_desc_dict['社保纠纷'] = '某年某月某日，到某处工作，双方签订/未签订劳动合同，约定工资x元，公司某年某月某日开始办理/未办理社会保险（是否存在未足额缴纳保险费及非因本人意愿中断的情形）。'
# problem_desc_dict['继承问题'] = '某某死后留有xx遗产（是否有遗嘱/扶养协议），某某拥有继承权，继承人尽/未尽到赡养义务的具体情形。'
# problem_desc_dict['借记卡纠纷'] = '向某银行账户存储x元，发现账户内存款被盗刷或减少，银行存在xx过错（本人是否报警或挂失）。'
# problem_desc_dict['保管合同纠纷'] = 'x年x月x日，x将物品交给x保管，签订/未签订保管合同，约定/未约定保管费，保管人未尽保管义务导致保管物毁损或未按时归还、委托人拖欠保管费等（请具体描述违约行为及后果）。'
# problem_desc_dict['保证合同纠纷'] = 'x年x月x日，x向x提供保证，签订/未签订保证合同，约定保证范围为x，保证方式为x，保证期限届满，x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）。'
# problem_desc_dict['承揽合同纠纷'] = 'x年x月x日，x委托x加工定做x，双方签订/未签订承揽加工合同，约定费用x元，一方履行/未履行合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为（如：没有达到维修标准、存在质量问题等），造成x后果（请具体描述违约行为及后果）'
# problem_desc_dict['定金合同纠纷'] = 'x年x月x日，为x支付了定金x元，x违反约定导致合同目的无法实现（请具体描述违约行为及后果）。'
# problem_desc_dict['借用合同纠纷'] = 'x年x月x日，x将物品借给x使用，双方签订/未签订借用合同，约定/未约定借用费，x违反约定导致借用物毁损或借用人未按时归还借用物/拖欠借用费等（请具体描述违约行为及后果）。'
# problem_desc_dict['金融借款合同纠纷'] = 'x年x月x日，双方签订/未签订金融借款合同纠纷，借款的用途为x，出借人交付/未交付款项，借款利率利息为x%，逾期利息为x%，双方存在x违约行为（请具体描述过程，如未按约定还款、未按约偿还借款本息等）。'
# problem_desc_dict['居间合同纠纷'] = 'x年x月x日，双方签订/未签订居间合同，约定x向x提供x居间服务，居间方履行/未履行x居间义务导致x委托事项完成/未完成，委托方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）。'
# problem_desc_dict['农林牧渔承包合同纠纷'] = 'x年x月x日，x承包了x，双方签订/未签订承包合同（协议），承包方支付/未支付承包费，发包方交付/未交付约定的土地/林地/池塘，及合同履行过程中x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）。'
# problem_desc_dict['委托合同纠纷'] = 'x年x月x日，x委托x做x事，双方签订/未签订委托合同，一方履行/未履行合同，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）。'
# problem_desc_dict['物业服务合同纠纷'] = 'x年x月x日，双方签订/未签订物业服务合同，物业公司提供履行/未提供物业服务，业主支付/未支付物业费，及合同履行过程中x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）。'
# problem_desc_dict['信用卡纠纷'] = 'x年x月x日，x向x银行申请办理信用卡，银行发放/未发放信用卡，x使用/未使用该信用卡消费，或透支/未透支使用，如期/未如期偿还，或存在其他行为造成x后果（请具体描述过程）。'
# problem_desc_dict['赠与合同纠纷'] = 'x年x月x日，签订/口头约定赠与合同，赠与物为x，赠与人交付/未交付赠与物，x违反x义务（请具体描述违约行为及后果）。'
# problem_desc_dict['装饰装修合同纠纷'] = 'x年x月x日，签订/未签订装修装饰合同，装修方提供/未提供装修服务，另一方支付/未支付装修款，及其他违约行为（请具体描述违约行为及后果，如出现质量问题）。'
# problem_desc_dict['仓储合同纠纷'] = 'x年x月x日，签订/未签订仓储合同，一方当事人提供/未提供仓储服务，另一方支付/未支付费用，x存在x违约行为（请具体描述过程，如贬值、损坏、擅自移动、发生火灾等）'
# problem_desc_dict['缔约过失责任纠纷'] = 'x年x月x日，计划签订x合同，在订立合同的过程中一方违背了诚实信用原则作出x行为（请具体描述，如恶意进行磋商、不具备公开出售条件、提供虚假情况），合同成立/未成立，给另一方造成了x损失（请具体描述违约行为及后果）。'
# problem_desc_dict['典当纠纷'] = 'x年x月x日，签订/未签订典当合同，一方支付/未支付当金，典当期满，x逾期赎当/续当，支付/未偿还当金本息、综合费，x存在x违约行为（请具体描述过程，如没有赎当、无力还款等）。'
# problem_desc_dict['房屋拆迁安置补偿合同纠纷'] = 'x年x月x日，签订/未签订房屋拆迁安置补偿合同，约定的内容（如过渡费、补偿款、安置房等内容），x存在x违约行为（请具体描述过程，如房屋未交付、无法入住等）。'
# problem_desc_dict['广告合同纠纷'] = 'x年x月x日，签订/未签订广告合同，一方提供/未提供广告服务，另一方支付/未支付广告费，x存在x违约行为（请具体描述过程，如拒绝付款等）。'
# problem_desc_dict['合伙协议纠纷'] = 'x年x月x日，签订/未签订合伙协议，x全面履行出资义务/存在x违约行为，合伙期间债权债务情况（请具体描述自合伙开始至结束具体情形）。 '
# problem_desc_dict['建设工程分包合同纠纷'] = 'x年x月x日，双方签订/未签订建设工程分包合同，一方履行/未履行合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为（如：违法分包、没有资质等），造成x后果（请具体描述违约行为及后果）'
# problem_desc_dict['建设工程设计合同纠纷'] = 'x年x月x日，双方签订/未签订建设工程设计合同，一方履行/未履行合同合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）'
# problem_desc_dict['建设用地使用权合同纠纷'] = 'x年x月x日，签订/未签订建设用地使用权合同，一方履行/未履行合同义务，另一方支付/未支付费用，及合同履行过程中x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）。'
# problem_desc_dict['民间委托理财合同纠纷'] = 'x年x月x日，签订/未签订民间委托理财合同，x全面履行合同义务/存在x违约行为，在委托期限届满时，足额/未足额返还投资款、约定的收益，及其他违约行为（请具体描述过程，如拒绝付款、拒不归还、占为己有等）。'
# problem_desc_dict['农村土地承包合同纠纷'] = 'x年x月x日，x承包x土地，双方签订/未签订土地承包合同（协议），承包方支付/未支付承包费，发包方交付/未交付约定的土地，及合同履行过程中x违反约定存在x行为，造成x后果（请具体描述违约行为及后果）。'
# problem_desc_dict['施工合同纠纷'] = 'x年x月x日，签订/未签订建设施工合同，合同履行过程中x违反约定存在x行为（请具体描述，如拖欠施工款、未开工、质量缺陷等），给另一方造成x损失。'
# problem_desc_dict['服务合同纠纷'] = 'x年x月x日，x在x处办理x业务/会员卡/到某处消费等，x提供/未提供服务，x支付/未支付服务费，或x存在x违约行为（请具体描述过程，如私自切断信号、根本无法正常使用、额外扣费等），造成x损害后果。（请具体描述，如致使受伤、旅客财物被盗等）'
# problem_desc_dict['运输合同纠纷'] = 'x年x月x日，x与x签订运输合同/搭乘车辆，x履行/未履行运输义务，x支付/未支付货款，或x存在x违约行为，造成x损害后果。（请具体描述过程及结果，如突然刹车，与客车相撞等）'
# problem_desc_dict['名誉权纠纷'] = 'x年x月x日，x实施了x侵权行为（请具体描述过程，如诽谤、散布诽谤言论等），侵权人是否故意，对x造成了x损害后果。（请具体描述结果，如导致精神负担、造成经济损失等）'
# problem_desc_dict['肖像权纠纷'] = 'x年x月x日，x实施了x侵权行为（请具体描述过程，如用照片进行商业宣传），侵权人是否故意，对x造成了x损害后果。（请具体描述，如给精神造成损害等）'
# problem_desc_dict['姓名权纠纷'] = 'x年x月x日，x实施了x侵权行为（请具体描述过程，如冒名担保、盗用身份证复印件等），侵权人是否故意，对x造成了x损害后果。（请具体描述结果，如经济受损害、造成精神伤害等）'
# problem_desc_dict['一般人格权纠纷'] = 'x年x月x日，x实施了x侵权行为（请具体描述过程，如辱骂、骚扰等），侵权人是否故意，对x造成了x损害后果。（请具体描述结果，如经济严重损失、伤害身心健康等）'
# problem_desc_dict['隐私权纠纷'] = 'x年x月x日，x实施了x侵权行为（请具体描述过程，如泄露个人信息、窃取个人信息等），侵权人是否故意，对x造成了x损害后果。（请具体描述，如引发不良心理状态、造成精神伤害等）'
# problem_desc_dict['生命权、健康权、身体权纠纷'] = 'x年x月x日，x实施了x侵权行为（请具体描述过程，如砸伤他人等），侵权人是否故意，对x造成了x损害后果。（请具体描述结果，如蒙受经济损失、导致死亡等）'
# problem_desc_dict['产品责任纠纷'] = 'x年x月x日，购买了x产品，产品存在x问题，（请具体描述过程，如不符合国家安全标准、质量存在问题等），造成了x损害后果。（请具体描述结果，如财物毁损、造成严重烧伤等）'
# problem_desc_dict['提供劳务者致害责任纠纷'] = 'x年x月x日，签订/未签订劳务合同/协议，劳务人员在提供劳务期间存在x行为（请具体描述过程，如砸到他人等），造成x后果。（请具体描述结果，如骨折、致多人伤亡等）'
# problem_desc_dict['铁路运输人身损害责任纠纷'] = 'x年x月x日，签订运输合同/搭乘车辆，期间存在x违约行为（请具体描述，如将人摔伤、将人撞倒等），侵权人是否故意或重大过失（如未设警示、管理不善），造成x损害后果。（请具体描述结果，如骨折、高位截肢等）'
# problem_desc_dict['提供劳务者受害责任纠纷'] = 'x年x月x日，签订/未签订劳务合同/协议，在提供劳务期间x存在x行为（请具体描述过程，如被砸伤等），导致劳务人员造成x损害后果（请具体描述结果，如导致骨折、多处伤残等），侵权人是否故意或重大过失（如未严格制定操作规程、未尽到监管义务等）。'
# problem_desc_dict['用人单位责任纠纷'] = 'x年x月x日，签订/未签订劳动合同，劳动者在执行职务/非执行职务的过程中，存在x行为（请具体描述过程，如砸伤第三人等），对受害人造成x损害后果（请具体描述结果，如导致第三人死亡、骨折等）'
# problem_desc_dict['教育机构责任纠纷'] = 'x年x月x日，受害人在x学校接受教育，在校期间，遭遇了x行为（请具体描述过程，如被打伤、被猥亵等），侵权人是否故意或重大过失（如看护不力、未尽管理职责等），对受害人造成了x损害后果。（请具体描述结果，如心理障碍、造成身心受到伤害等）'
# problem_desc_dict['监护人责任纠纷'] = 'x年x月x日，x对受害人实施了x行为（请具体描述过程，如戳伤、撞倒等），造成了x损害后果。（请具体描述结果，如骨折、全身多处擦伤等）'
# problem_desc_dict['公共场所管理人责任纠纷'] = 'x年x月x日，在x地方（如：餐馆、宾馆等公共场所），遭受了x损害（请具体描述过程，如遭到殴打、被捅死等），管理人存在x过错（如安全警示标志设立不规范、未尽监督管理责任等）。（请具体描述结果，如粉碎性骨折等）'
# problem_desc_dict['义务帮工人受害责任纠纷'] = 'x年x月x日，x为x提供义务帮工，被帮工人明确/未明确拒绝，帮工人收取/未收取报酬，因x原因导致帮工人受到x伤害。（请具体描述过程，如致使物品坠落、被物品砸伤等）'
# problem_desc_dict['医疗损害责任纠纷'] = 'x年x月x日，x在医院接受治疗，因医院x行为（请具体描述过程，如诊治失误、销毁病历等），造成x损害后果。（请具体描述结果，如留下后遗症、病情加重等）'
# problem_desc_dict['物件脱落、坠落损害责任纠纷'] = 'x年x月x日，x在x场所，因物件脱落/坠落造成x损害后果（请具体描述过程及结果，如高空掉下某物被砸伤等），管理人是否尽到管理义务（如提醒、设置标志等）。'
# problem_desc_dict['建筑物、构筑物倒塌损害责任纠纷'] = 'x年x月x日，x在x场所，因建筑物/构筑物倒塌（请具体描述过程，如断裂、坍塌、被砸伤等），对受害人造成x损害后果（请具体描述结果，如生活不能自理、当场死亡等），管理人是否尽到管理义务（如设置防护栏、设置提醒标志等）。'
# problem_desc_dict['公共道路妨碍通行损害责任纠纷'] = 'x年x月x日，x在x道路，因公共道路妨碍通行（请具体描述过程，如在路面堆放物品导致车辆侧翻等），造成x损害结果（请具体描述结果，如致伤残、脑震荡等）,道路管理者是否尽到管理提醒义务。'
# problem_desc_dict['地面施工、地下设施损害责任纠纷'] = 'x年x月x日，x在x道路，因地面施工/地下设施（请具体描述过程），造成x损害结果（请具体描述结果，如致使翻车、导致爆胎等）,道路管理者是否尽到管理提醒义务（请具体描述，如未采取措施、没有标志等）。'
# problem_desc_dict['触电人身损害责任纠纷'] = 'x年x月x日，x因触电（请具体描述过程，如被电倒、遭到电击等），造成x损害结果（请具体描述后果，如被电晕、被电伤等）。'
# problem_desc_dict['饲养动物致人损害责任纠纷'] = 'x年x月x日，x被x动物伤害（请具体描述过程，如被撞倒、猛咬等），造成x损害后果（请具体描述结果，如狂犬病、骨折等），动物饲养人是否尽到饲养义务，受害者是否存在故意挑逗动物行为（请具体描述）。'
# problem_desc_dict['网络侵权责任纠纷'] = 'x年x月x日，x通过网络进行了x行为（请具体侵权行为，如发布贬损言论、散布负面帖文等），对x造成了x损害后果。（请具体描述结果，如造成商誉降低等）'
# problem_desc_dict['宅基地使用权纠纷'] = 'x年x月x日，是/不是该村村民，通过x方式取得该宅基地使用权（如：本村分配、继承、交换等），在此过程中x存在x行为影响到该宅基地的使用。（请具体描述过程及结果，如影响正常居住等）'
# problem_desc_dict['占有保护纠纷'] = 'x是/不是该物的所有人或通过x方式对该物享有x权利（如：通过抵押取得抵押权），x年x月x日，该物被x侵权人非法占有拒不归还或受到x损害导致了x后果。（请具体描述过程及结果，如擅自出租、擅自占用等）'
# problem_desc_dict['承包地征收补偿费用分配纠纷'] = 'x年x月x日，x承包地被征收，补偿款为x元，通过x种方式取得该承包地经营权，承包地被征收时，由于x原因未足额分得应得补偿款或存在x争议。（请具体描述过程及结果）'
# problem_desc_dict['抵押权纠纷'] = 'x年x月x日，为何事签订/未签订抵押合同，办理/未办理抵押登记，抵押期限x个月/年，抵押期限届满，债务清偿/未清偿完毕或满足/不满足抵押权实现条件。（请具体描述过程及结果）'
# problem_desc_dict['物权保护纠纷'] = 'x是/不是该物的所有人或通过x方式对该物享有x权利（如：通过抵押取得抵押权），x年x月x日，该物被x侵权人非法占有拒不归还或受到x损害导致了x后果。（请具体描述过程及结果，如拒不归还等）'
# problem_desc_dict['侵害集体组织成员权益纠纷'] = 'x是/不是本集体经济组织成员（村民），在本集体经济组织（本村）给成员（村民）分配权益时，存在/不存在不公平分配情形，造成x损害后果。（请具体描述过程及结果）'
# problem_desc_dict['相邻关系纠纷'] = 'x与x存在/不存在相邻关系(如：房屋、土地、道路相邻），x存在x行为（请具体描述过程，如擅自改建、排放粪便等），给另一方当事人造成x损害后果。（请具体描述结果，如影响排水、对日照构成影响等）'
# problem_desc_dict['共有纠纷'] = 'x物属于/不属于x与x的共有物（请具体描述共有的原因，如：夫妻共有、家庭共有、共同继承、合伙等），存在/不存在协议，现因x原因请求确认享有共有权/分割共有物（请具体描述过程，如分家、离婚等）。'
# problem_desc_dict['农村土地承包经营权纠纷'] = 'x年x月x日，通过x种方式承包x村土地/林地，取得/未取得该承包地经营权，现该地存在x情形，导致该地无法正常使用。（请具体描述被侵权行为的过程及结果，如被非法侵占或收回）'
# problem_desc_dict['建设用地使用权纠纷'] = 'x年x月x日，通过x种方式（请具体描述过程，如：划拨、出让或转让）取得x建设用地使用权，签订/未签订建设用地使用权合同，办理/未办理相关手续（如：审批、变更、注销），由于x行为导致x后果。（请具体描述过程及结果，被侵犯x合法权益造成该建设用地及其建筑物无法正常使用）'


raw_path = '/datadisk2/tyin/raw/'  # 源数据存储路径，不同模型共享

config_path = '../data/inference_with_reason/config/'  # 配置文件路径
if not os.path.exists(config_path):
    os.mkdir(config_path)

model_path = '../model_files/inference_with_reason/'  # 模型文件路径
if not os.path.exists(model_path):
    os.mkdir(model_path)

LTP_DATA_DIR = '../data/inference_with_reason/ltp'  # ltp模型目录的路径
seg_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性模型的路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

# 加载模型
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(seg_model_path, model_path + 'userdict.txt')  # 加载模型
postagger = Postagger()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型
parser = Parser()  # 初始化实例
parser.load(par_model_path)


##############################################################################################################################################
#
# 重复数据处理
#
##############################################################################################################################################

def column_as_string(data):
    """
    将数据格式转为string
    :param data: 数据，格式为pandas.DataFrame
    :return: 新DataFrame
    """
    for column in data.columns:
        data[column] = data[column].astype(str)
    return data


def word_filter(data, column, filter_words):
    """
    将指定列中包含过滤词的数据过滤掉
    :param data: 数据，格式为pandas.DataFrame
    :param column: 列名
    :param filter_words: 过滤词
    :return: 新DataFrame
    """

    def _match(x):
        result = []
        for t in re.split('[。，；：,;:]', x):
            result.append(1 if len(re.findall(word, t)) == 0 else 0)
        return bool(min(result))

    for word in filter_words:
        data = data[data[column].apply(lambda x: _match(x))]
    return data


poses = {'n', 'v', 'p', 'd', 'nh', 'q', 'a', 'b', 'wp'}


def pos_filter(content):
    words = list(segmentor.segment(content))
    postags = list(postagger.postag(words))
    content = ''.join([w for i, w in enumerate(words) if postags[i] in poses])
    content = re.sub('[，。；：][，。；：]', '，', content)
    if content[0] in ['，', '。', '；', '：']:
        content = content[1:]
    return content


def chinese_filter(content):
    return ''.join([t for t in re.findall('[\u4E00-\u9FA5]', content)])


def html_clean(html_string):
    html_string = html_string.replace('(', '（').replace(')', '）').replace(',', '，').replace(':', '：').replace(';',
                                                                                                              '；').replace(
        '?', '？').replace('!', '！')
    html_string = re.sub('\d，\d', lambda x: x.group(0).replace('，', ''), html_string)
    html_string = html_string.replace('</a>。<a target=', '</a>、<a target=')
    while len(re.findall('(<a target=.*?>(.*?)</a>)', html_string)) > 0:
        a = re.findall('(<a target=.*?>(.*?)</a>)', html_string)
        html_string = html_string.replace(a[0][0], a[0][1])
    html_string = html_string.replace('&times；', 'x').replace('&hellip；', '…').replace('＊', 'x').replace('*', 'x')
    html_string = html_string.replace('&ldquo；', '“').replace('&rdquo；', '”')
    html_string = html_string.replace('&lt；', '<').replace('&gt；', '>')
    html_string = html_string.replace('&permil；', '‰')
    return html_string


def content_extract(original_sentence, yuangao=None, beigao=None):
    """
    过滤非中文，原告诉称这种特殊字段，过滤原告、被告名字
    :param row: DataFrame行数据
    :return:
    """
    # 过滤非中文
    sentences = re.sub('[^\u4E00-\u9FA5]', '', original_sentence)
    # 过滤开头的固定性语句
    sentences = re.sub('(原告.{0,6}诉称|原告.{0,6}向本院提出.{0,6}诉讼请求)', '', sentences)
    # 过滤原告
    if yuangao == yuangao and yuangao is not None:
        sentences = sentences.replace(yuangao, '')
    # 过滤被告
    if beigao == beigao and beigao is not None:
        sentences = sentences.replace(beigao, '')
    return sentences


def repeat_filter(data, column, extra_columns=None):
    """
    column剔除非中文以及原告被告姓名后，重复内容删除
    :param data: 数据，格式为pandas.DataFrame
    :return: 新DataFrame
    """

    def _extract(row):
        yuangao = row['yuangao'] if 'yuangao' in row else None
        beigao = row['beigao'] if 'beiago' in row else None
        return content_extract(row[column], yuangao, beigao)

    if len(data) == 0:
        return data
    data['temp'] = data.apply(_extract, axis=1)
    columns = ['temp'] if extra_columns is None else ['temp'] + extra_columns
    data = data.drop_duplicates(subset=columns)
    data = data.drop('temp', axis=1)
    return data


def question_transfer(question):
    question = json.loads(question) if question == question and question.startswith('{') else {}
    question = {k: v for k, v in question.items()}
    result = []
    for k, v in question.items():
        if v == '是':
            result.append(k.replace('是否', '').replace('？', ''))
        elif v == '否':
            result.append(
                k.replace('是否存在', '不存在').replace('是否具有', '不具有').replace('是否属于', '不属于').replace('是否有', '没有').replace(
                    '是否', '没有').replace('？', ''))
        elif isinstance(v, str) and v != '以上都不是' and v != '以上都没有':
            result.append(v)
        elif isinstance(v, list) and '以上都不是' not in v and '以上都没有' not in v:
            result.append('。'.join(v))
    return '。'.join(result)


##############################################################################################################################################
#
# 同义词扩充
#
##############################################################################################################################################

# 同义词典
with open(model_path + '同义词.json', 'r') as f:
    sim_dict = json.load(f)


def tongyici_generate():
    data = pd.read_csv('./tongyici.csv')
    dic = {v[0]: v[1].split(',') for v in data.values}
    with open(model_path + '同义词.json', 'w') as f:
        json.dump(dic, f)


def keyword_drop_duplicates(keyword_list):
    """
    剔除关键词列表里具有包含关系的关键词
    :param keyword_list:
    :return:
    """
    result_list = list(keyword_list)
    for i, k1 in enumerate(keyword_list):
        if k1 not in result_list or len(re.findall('(不|未|没有|无|非)', k1)) > 0:
            continue
        for j, k2 in enumerate(keyword_list[i + 1:]):
            if k2 not in result_list or len(re.findall('(不|未|没有|无|非)', k1)) > 0:
                continue
            if len(re.findall(k1, k2)) > 0:
                result_list.remove(k2)
            elif len(re.findall(k2, k1)) > 0:
                result_list.remove(k1)
                break
    return result_list


def keyword_split(keyword):
    """
    对关键词进行切分
    :param keyword:
    :return:
    """
    result_list = []
    r = ""
    for w in keyword:
        if len(re.findall('[\u4E00-\u9FA5]', w)) > 0:
            r += w
        else:
            if len(r) > 0:
                result_list.append(r)
                r = ""
            result_list.append(w)
    if len(r) > 0:
        result_list.append(r)
    result = []
    for r in result_list:
        result += jieba.lcut(r)
    return result


def keyword_list_expand(keyword_list):
    """
    基于同义词表扩充关键词, 保持原关键词排序一致
    :param keyword_list:
    :return:
    """
    if keyword_list != keyword_list or keyword_list is None:
        return keyword_list
    return_str = False
    if isinstance(keyword_list, str):
        keyword_list = [keyword_list]
        return_str = True
    result_list = []
    for keyword in keyword_list:
        ks = keyword_split(keyword)
        rs = list(ks)
        for i, k in enumerate(ks):
            if k in sim_dict:
                rs[i] = '(' + '|'.join([k] + sim_dict[k]) + ')'
        result_list.append(''.join(rs))
    if return_str:
        result_list = result_list[0]
    return result_list


##############################################################################################################################################
#
# 规则扩充
#
##############################################################################################################################################

def rule_match_expand(id, sequence, factor_keywords, problem, suqiu):
    """
    返回匹配到的特征list
    :param id: 事实id
    :param sequence: 未匹配到特征的事实描述
    :param factor_keywords: 特征关键词list
    :param problem : 问题类型
    :param suqiu : 诉求
    :return: list
    """

    def keywords_add(patten_list, return_list, fact_need, is_clean=False):
        patten_need = '|'.join(patten_list)
        if len(re.findall(patten_need, sequence)) > 0:
            if is_clean:
                return_list.clear()
            return_list.append(fact_need)

    factor_keywords_list = list(factor_keywords.index)
    factor_keywords_str = '|'.join(factor_keywords_list)
    # print('aaa', factor_keywords)

    return_list = []
    if suqiu == '离婚':
        patten_list_1 = ['第三者', '小三', '离家出走', '出走.*至今未归', '大.{0,1}出手', '同意离婚',
                         '(男|女|双)方同意离婚', '私奔', '感情不和', '出轨', '自愿离婚', '老婆跑了', '老公跑了', '男方跑了',
                         '女方跑了', '协议离婚', '签订.{0,6}离婚协议', '非自愿结婚', '他人同居', '感情不(和|合)', '性格不合',
                         '骗婚', '分居三年', '分开三年', '酗酒', '赌博', '吸毒', '外遇', '打我', '分居.{0,3}多年']
        fact_need_1 = '准予离婚的情形'
        keywords_add(patten_list_1, return_list, fact_need_1)

        patten_list_2 = ['没有.{0,6}结婚证', '未领证', '(未|没有|没)登记结婚', '(未|没有|没)领证',
                         '(未|没有|没)领结婚证', '(没有|没)结婚证', '(未|没有|没)结婚']
        fact_need_2 = '不准予离婚'
        keywords_add(patten_list_2, return_list, fact_need_2, is_clean=True)

        patten_list_3 = ['分居.{0,3}一年', '分居.{0,3}两年', '不愿意', '不同意离婚']
        fact_need_3 = '不准予离婚'
        keywords_add(patten_list_3, return_list, fact_need_3)

    if problem == '婚姻家庭':
        patten_list_5 = ['同居.{0,6}小.{0,1}孩']
        fact_need_5 = '婚姻关系'
        keywords_add(patten_list_5, return_list, fact_need_5)

        if suqiu == '婚姻无效':
            patten_list_7 = ['(未|没有|没)登记结婚', '(未|没有|没)领证', '(未|没有|没)领结婚证', '(没有|没)结婚证',
                             '(未|没有|没)结婚', '身份证.{0,10}假', '他哥.{0,6}身份证', '表(兄|妹)']
            fact_need_7 = '婚姻无效事由'
            keywords_add(patten_list_7, return_list, fact_need_7, is_clean=True)

        patten_list_4 = ['登记完婚', '存在婚姻', '夫妻之实']
        fact_need_4 = '婚姻关系'
        keywords_add(patten_list_4, return_list, fact_need_4)

        if '婚姻关系' not in return_list:
            fact_need_6 = '婚姻无效'
            patten_list_6 = ['同居']
            flag = 0
            if '婚姻无效' not in return_list:
                flag = 1
            keywords_add(patten_list_6, return_list, fact_need_6)
            if '婚姻无效' not in return_list:
                flag = 0
            # if flag:
            #     print(sequence)
            #     print('nnn', return_list)

        patten_list_8 = ['拒绝.{0,6}支付.{0,6}抚养费', '不给.*生活费', '(未出|不出).*抚养费', '不管孩子']
        fact_need_8 = '未支付抚养费'
        keywords_add(patten_list_8, return_list, fact_need_8)

        patten_list_9 = ['子女.{0,6}问题.{0,6}约定', '抚养协议']
        fact_need_9 = '存在抚养费协议或判决'
        keywords_add(patten_list_9, return_list, fact_need_9)

        patten_list_11 = ['(儿子|女儿|孩子|小孩).{0,6}个月', '(儿子|女儿|孩子|小孩)一岁', '(儿子|女儿|孩子|小孩)两岁']
        fact_need_11 = '优先抚养权'
        keywords_add(patten_list_11, return_list, fact_need_11)

        patten_list_13 = ['共有房产']
        fact_need_13 = '共同(共有)财产'
        keywords_add(patten_list_13, return_list, fact_need_13)

        patten_list_20 = ['常年在外', '没付.*抚养费']
        fact_need_20 = '未履行扶养义务'
        keywords_add(patten_list_20, return_list, fact_need_20)

        patten_list_21 = ['没.{0,6}（探视|探望）规则']
        fact_need_21 = '探望权不确定'
        keywords_add(patten_list_21, return_list, fact_need_21)

        patten_list_22 = ['支付抚养费', '约定.+?(探望|探视)', '有探视权']
        fact_need_22 = '享有探望权'
        keywords_add(patten_list_22, return_list, fact_need_22)

        patten_list_23 = ['(探望|探视)困难', '电话不接', '不接电话']
        fact_need_23 = '探望权受阻'
        keywords_add(patten_list_23, return_list, fact_need_23)

        patten_list_90 = ['收养登记']
        fact_need_90 = '收养关系'
        keywords_add(patten_list_90, return_list, fact_need_90)

        patten_list_89 = ['欠.*债', '借.*钱', '债务']
        fact_need_89 = '共同债务'
        keywords_add(patten_list_89, return_list, fact_need_89)

        patten_list_88 = ['离婚', '感情(破裂|不合)']
        fact_need_88 = '解除婚姻关系'
        keywords_add(patten_list_88, return_list, fact_need_88)

        patten_list_87 = ['车', '存款', '房', '辆']
        fact_need_87 = '共同财产'
        keywords_add(patten_list_87, return_list, fact_need_87)

        patten_list_86 = ['婚前.*房']
        fact_need_86 = '个人财产'
        keywords_add(patten_list_86, return_list, fact_need_86)

        patten_list_85 = ['不.*赡养', '不给.*生活费', '老人.*没人管']
        fact_need_85 = '未尽赡养义务'
        keywords_add(patten_list_85, return_list, fact_need_85)

        patten_list_84 = ['婚前.*房']
        fact_need_84 = '个人财产'
        keywords_add(patten_list_84, return_list, fact_need_84)

        patten_list_83 = ['未尽到抚养义务']
        fact_need_83 = '变更监护权的法定事由'
        keywords_add(patten_list_83, return_list, fact_need_83)

        if suqiu == '探望权':
            return_list.append('享有探望权')
    # print(return_list)

    # if suqiu == '存在劳动关系':
    #     print(factor_keywords['不存在劳动关系'])
    if problem == '借贷纠纷':
        # if suqiu == '支付利息':
        #     print(factor_keywords['不支付利息'])

        patten_list_10 = ['本金利息无法按时支付', '逾期']
        fact_need_10 = '到期未清偿借款'
        keywords_add(patten_list_10, return_list, fact_need_10)

        patten_list_12 = ['转账记录', '录音', '转账凭证', '聊天记录', '(支付宝|微信)转账', '合同', '收据', '协议', '收条', '借款条子']
        fact_need_12 = '证据'
        keywords_add(patten_list_12, return_list, fact_need_12)

        patten_list_14 = ['盗用.{0,6}身份证', '(威胁|强制).{0,6}借(钱|款)', '身份证泄露', '断头贷', '砍头息']
        fact_need_14 = '借贷行为无效'
        keywords_add(patten_list_14, return_list, fact_need_14)

        patten_list_15 = ['到期.{0,3}还款', '合同', '网络借贷', '借.{0,5}给', '个人借贷', '(借|贷|欠).*(千|万)', '出借', '朋友借', '借.*钱']
        fact_need_15 = '存在借贷关系'
        keywords_add(patten_list_15, return_list, fact_need_15)

        patten_list_99 = ['没.{0,5}打款', '到期偿还']
        fact_need_99 = '合同解除情形'
        keywords_add(patten_list_99, return_list, fact_need_99)

        patten_list_98 = ['借钱不还', '(未|不|没).{0,5}还', '未偿还本(金|息)', '未还款', '未清偿', '没有给', '还不了', '逾期', '(联系|找)不到人',
                          '没有任何消息']
        fact_need_98 = '到期未清偿借款'
        keywords_add(patten_list_98, return_list, fact_need_98)

        patten_list_97 = ['实际到账', '支付', '转账', '借.{0,10}元']
        fact_need_97 = '实际给付借款'
        keywords_add(patten_list_97, return_list, fact_need_97)

        patten_list_96 = ['年息', '月.{0,2}息']
        fact_need_96 = '借期内利息'
        keywords_add(patten_list_96, return_list, fact_need_96)

        if '证据' in return_list:
            return_list.append('存在借贷关系')
        if '实际给付借款' in return_list:
            return_list.append('存在借贷关系')

    if problem == '劳动纠纷':
        patten_list_16 = ['合同到期']
        fact_need_16 = '不存在劳动关系'
        keywords_add(patten_list_16, return_list, fact_need_16)

        patten_list_17 = ['拖欠工资', '裁员']
        fact_need_17 = '解除劳动关系'
        keywords_add(patten_list_17, return_list, fact_need_17)

        patten_list_69 = ['合同没到期', '劳动合同']
        fact_need_69 = '劳动关系'
        keywords_add(patten_list_69, return_list, fact_need_69)

        patten_list_68 = ['公司', '单位']
        fact_need_68 = '主体资格'
        keywords_add(patten_list_68, return_list, fact_need_68)

        patten_list_67 = ['工资']
        fact_need_67 = '财产关系'
        keywords_add(patten_list_67, return_list, fact_need_67)

        patten_list_66 = ['在.*岗位', '(去|到|招入).{0,10}工作']
        fact_need_66 = '人身关系'
        keywords_add(patten_list_66, return_list, fact_need_66)

        patten_list_65 = ['怀孕', '住院', '调休', '节假日', '交通事故']
        fact_need_65 = '用人单位违法解除劳动合同的情形'
        keywords_add(patten_list_65, return_list, fact_need_65)

    if problem == '交通事故':
        patten_list_18 = ['事故责任.{0,3}认定', '对方(不|拒绝)赔偿', '追尾', '交通事故责任书判定对方全责', '全责', '主责', '次责', '主要责任', '全部责任', '五五开']
        fact_need_18 = '交通事故责任认定'
        keywords_add(patten_list_18, return_list, fact_need_18)

        patten_list_19 = ['保险']
        fact_need_19 = '投保保险'
        keywords_add(patten_list_19, return_list, fact_need_19)

        patten_list_70 = ['撞', '追尾', '伤', '剐蹭', '刮', '擦']
        fact_need_70 = '造成损害结果'
        keywords_add(patten_list_70, return_list, fact_need_70)

    if problem == '工伤赔偿':
        patten_list_30 = ['(去|到).{0,10}工作']
        fact_need_30 = '人身关系'
        keywords_add(patten_list_30, return_list, fact_need_30)

        patten_list_92 = ['劳动合同']
        fact_need_92 = '劳动关系'
        keywords_add(patten_list_92, return_list, fact_need_92)

        patten_list_91 = ['工资']
        fact_need_91 = '财产关系'
        keywords_add(patten_list_91, return_list, fact_need_91)

    if problem == '买卖纠纷':
        patten_list_95 = ['违约', '逾期', '被骗', '对方不给', '不支付']
        fact_need_95 = '合同解除事由'
        keywords_add(patten_list_95, return_list, fact_need_95)

        patten_list_94 = ['合同', '网.{0,5}(购|买)']
        fact_need_94 = '存在合同关系'
        keywords_add(patten_list_94, return_list, fact_need_94)

        patten_list_93 = ['被骗', '让.*骗', '隐瞒合同内容', '欺骗']
        fact_need_93 = '合同无效'
        keywords_add(patten_list_93, return_list, fact_need_93)

        patten_list_77 = ['不按约定', '超时', '违约']
        fact_need_77 = '违约行为'
        keywords_add(patten_list_77, return_list, fact_need_77)

        if '合同解除事由' in return_list:
            return_list.append('存在合同关系')

    if problem == '继承问题':
        patten_list_82 = ['欠']
        fact_need_82 = '存在债务'
        keywords_add(patten_list_82, return_list, fact_need_82)

        patten_list_81 = ['遗属']
        fact_need_81 = '订立遗嘱'
        keywords_add(patten_list_81, return_list, fact_need_81)

        patten_list_80 = ['抚血金']
        fact_need_80 = '非遗产范围'
        keywords_add(patten_list_80, return_list, fact_need_80)

        patten_list_79 = ['同居']
        fact_need_79 = '不享有继承权'
        keywords_add(patten_list_79, return_list, fact_need_79)

        patten_list_78 = ['母亲', '父亲', '父母', '妈妈', '爸爸', '父母']
        fact_need_78 = '非遗产范围'
        keywords_add(patten_list_78, return_list, fact_need_78)

        return_list = [x for x in return_list if x in factor_keywords_list]

        # if len(return_list) == 0:
        #     with open('./data/marking.csv', 'a') as ff:
        #         sequence = sequence.replace(',', '，')
        #         need_data = str(id) + ',' + problem + ',' + suqiu + ',' + sequence + ',' + factor_keywords_str
        #         need_data = need_data.replace('\n', '')
        #         need_data = need_data.replace(' ', '')
        #         need_data = need_data.replace('\t', '')
        #         need_data = need_data.replace('\r', '')
        #         ff.write(need_data)
        #         ff.write('\n')

        return_list = list(set(return_list))
    return return_list


##############################################################################################################################################
#
# 裁判文书结构化信息提取
#
##############################################################################################################################################

##############################################################################################################################################
#
# 文书头尾信息数据提取
#
##############################################################################################################################################


def head_information_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    # 法院
    court = re.findall('<p align="center">(.*?)</p>', informations[0])[0]

    # 文书类型
    types = re.findall('<p align="center">(.*?)</p>', informations[1])[0]

    # 文书编号
    number = re.findall('<p align="right">(.*?)</p>', informations[2])[0]

    judger = None
    date = None
    clerk = None
    for info in informations[-3:]:
        if len(re.findall('<p>(.*?)</p>', info)) > 0:
            info = re.findall('<p>(.*?)</p>', info)[0]
            # 审判员
            for pattern in ['代审判员', '见习审判员', '代理审判员', '助理审判员', '审判员', '代审判长', '见习审判长', '代理审判长', '审判长', '助理审判长']:
                if info.startswith(pattern):
                    judger = info.replace('</br>', '；')
                    break

            # 时间
            if (info.startswith('一') or info.startswith('二')) and len(re.findall('[一二][，：、。；]', info)) == 0:
                date = info

            # 书记员
            for pattern in ['代书记员', '见习书记员', '代理书记员', '书记员']:
                if info.startswith(pattern):
                    clerk = info
    return court, types, number, judger, date, clerk


##############################################################################################################################################
#
# 提取原被告信息
#
##############################################################################################################################################


def participant_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('（[^。！？：<>]*?）', '', html_string)

    if html_string.startswith('<p>'):
        html_string = html_string[3:]
    if '</p>' in html_string:
        html_string = html_string[:html_string.index('</p>')]

    if len(re.findall(introduction_pattern, html_string)) > 0:
        pattern = re.findall(introduction_pattern, html_string)[0]
        html_string = html_string[:html_string.index(pattern)]
    elif len(re.findall(sucheng_pattern1, html_string)) > 0:
        patterns = re.findall(sucheng_pattern1, html_string)
        end_index = min([html_string.index(p) for p in patterns])
        html_string = html_string[:end_index]
    else:
        for pattern in renwei_pattern:
            if len(re.findall(pattern, html_string)) > 0:
                p = re.findall(pattern, html_string)[0][0]
                html_string = html_string[:html_string.rindex(p)]
                break

    participant = []
    yuangao = []
    beigao = []
    if '</br>' in html_string:
        infos = html_string.replace('</br>原告', '</br>###原告').replace('</br>被告', '</br>###被告')
    else:
        infos = html_string.replace('。原告', '。</br>###原告').replace('。被告', '。</br>###被告')
        infos = infos.replace('。负责', '。</br>负责')
        infos = infos.replace('。法定', '。</br>法定')
        infos = infos.replace('。委托', '。</br>委托')
    for info in re.split('</br>###', infos):
        if '纠纷' in info:
            continue
        person = []
        for p in info.split('</br>'):
            for i in ['告', '人', '者', '代理', '代表', '业主']:
                if i not in p: continue
                iden = p[:p.index(i) + len(i)]
                p = p[len(iden):]
                if len(p) == 0: continue
                if p[0] in ['，', '。', '：', '；']:
                    p = p[1:]
                name = re.split('[，；。]', p)[0]
                desp = p[len(name) + 1:]
                person.append([iden, name + '。' + desp])
                if iden in ['原告', '起诉人']:
                    yuangao.append(name)
                elif iden in ['被告', '第一被告']:
                    beigao.append(name)
                break
        if len(person) > 0:
            participant.append(person)
    return participant if len(participant) > 0 else None, \
           yuangao if len(yuangao) > 0 else None, \
           beigao if len(beigao) > 0 else None


##############################################################################################################################################
#
# 提取简介信息
#
##############################################################################################################################################

introduction_pattern = '一案|两案|二案|\d案|纠纷案'


def introduction_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('（[^。！？：<>]*?）', '', html_string)
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)

    introduction = {}
    if len(re.findall(introduction_pattern, html_string)) > 0:
        pattern = re.findall(introduction_pattern, html_string)[0]
        start_index = html_string.index(pattern)
        while html_string[start_index] not in ['，', '。', '；', '：', '>', '！', '？']:
            start_index -= 1
        html_string = html_string[start_index + 1:]

        if '</p>' in html_string:
            html_string = html_string[:html_string.index('</p>')]
        if '</br>' in html_string:
            html_string = html_string[:html_string.index('</br>')]

        if len(re.findall(sucheng_pattern1, html_string)) > 0:
            patterns = re.findall(sucheng_pattern1, html_string)
            end_index = min([html_string.index(p) for p in patterns])
            html_string = html_string[:end_index]
        else:
            for pattern in renwei_pattern:
                if len(re.findall(pattern, html_string)) > 0:
                    p = re.findall(pattern, html_string)[0][0]
                    html_string = html_string[:html_string.rindex(p)]
                    break

        infos = html_string
        introduction['简介'] = infos
        if len(re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日).{0,3}提起诉讼', infos)) > 0:
            introduction['诉讼日期'] = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日).{0,3}提起诉讼', infos)[0]
        if len(re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日).{0,2}受理', infos)) > 0:
            introduction['立案日期'] = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日).{0,2}受理', infos)[0]
        if len(re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日).{0,2}开庭', infos)) > 0:
            introduction['开庭日期'] = re.findall(r'(\d{4}年\d{1,2}月\d{1,2}日).{0,2}开庭', infos)[0]
        if len(re.findall('(公开|开庭|依法).*(审理|审了).*?[，。]([^。未不没]*参加.{0,1}诉讼)', infos)) > 0:
            introduction['到庭情况'] = re.findall('(公开|开庭|依法).*(审理|审了).*?[，,。]([^。未不没]*参加.{0,1}诉讼)', infos)[0][2]

    return introduction if len(introduction) > 0 else None


##############################################################################################################################################
#
# 原告诉称提取诉请和事实
#
##############################################################################################################################################

suqiu_pattern = [
    '((请求|要求|申请|诉请)[^，。；：不]*(判令|判决|改判|裁决|确认|判准)：)',
    '(请[^，。；：不]*(判令|判决|改判|裁决|确认|判准)：)',
    '(([现故。；]|起诉|诉至|诉讼至|诉诸|诉请|为此|据此|因此|综上|为维护.{0,3}合法权益).{0,6}(要求|请求)被告：)',
    '(([现故。；]|起诉|诉至|诉讼至|诉诸|诉请|为此|据此|因此|综上|为维护.{0,3}合法权益).{0,6}(要求|请求)：)',
    '((请求|要求|诉请|诉求|诉讼请求|依法判令).{0,2}：)',
    '((请求|要求|申请|诉请)[^，。；：不]*(判令|判决|改判|确认|判准)[^。]{2})',
    '((请|要求)[^，。；：不]*(判令|判决|改判|确认|判准)[^。]{2})',
    '((请|要求)[^。；：不]*(判令|判决|改判|确认|判准)[^。]{2})',
    '(([现故。；]|起诉|诉至|诉讼至|诉诸|诉请|为此|据此|因此|综上|为维护.{0,3}合法权益).{0,6}(要求|请求)[^。]{2})',
    '((请求|要求|申请|诉请)[^，。；：不]*裁决[^。]{2})',
    '((请|要求)[^，。；：不]*裁决[^。]{2})',
    '((请|要求)[^。；：不]*裁决[^。]{2})',
]
signifier_pattern = '[\d一二三四五六七八九][，、\.]'
signifier_pattern1 = '\d[，、\.]'
signifier_pattern2 = '[一二三四五六七八九][，、\.]'


def suqing_extract(sucheng):
    """
    将陈述中的诉求和事实分离，按照特殊的词进行分离
    :param suqing_sentences: 陈述
    """

    for p in ['事实.{0,1}理由：']:
        if len(re.findall(p, sucheng)) > 0:
            pattern = re.findall(p, sucheng)[0]
            suqiu = sucheng[:sucheng.index(pattern)]
            return suqiu

    if len(re.findall(signifier_pattern1, sucheng[:2])) > 0:
        end_index = 0
        while end_index < len(sucheng) - 1:
            if sucheng[end_index] == "。":
                if end_index < len(sucheng) - 2 and len(
                        re.findall(signifier_pattern1, sucheng[end_index + 1: end_index + 3])) == 0:
                    break
            end_index += 1
        if len(re.findall(signifier_pattern1, sucheng[2: end_index + 1])) > 0:
            return sucheng[:end_index + 1]

    if len(re.findall(signifier_pattern2, sucheng[:2])) > 0:
        end_index = 0
        while end_index < len(sucheng) - 1:
            if sucheng[end_index] == "。":
                if end_index < len(sucheng) - 2 and len(
                        re.findall(signifier_pattern2, sucheng[end_index + 1: end_index + 3])) == 0:
                    break
            end_index += 1
        if len(re.findall(signifier_pattern2, sucheng[2: end_index + 1])) > 0:
            return sucheng[:end_index + 1]

    for p in suqiu_pattern:
        if len(re.findall(p, sucheng)) > 0:
            pattern = re.findall(p, sucheng)[-1][0]
            start_index = sucheng.index(pattern)
            end_index = start_index + len(pattern)
            if sucheng[end_index:] in ['判决。', '裁决。', '确认。']:
                continue
            while start_index >= 0:
                if sucheng[start_index] in ['。', '，', '：', '；', '！', '？', '”', '“']:
                    break
                start_index -= 1
            while end_index < len(sucheng) - 1:
                if sucheng[end_index] == "。":
                    if end_index < len(sucheng) - 2 and len(
                            re.findall(signifier_pattern, sucheng[end_index + 1: end_index + 3])) == 0:
                        break
                end_index += 1
            return sucheng[start_index + 1: end_index + 1]
    return None


sucheng_pattern1 = [
    '原告[^。；未]*?诉称', '原告[^。；未]*?提出[^。；，撤]{0,5}请求', '起诉[^。；未]*?要求',
    '起诉[^。；未]*?认为', '原告[^。；未]*?称[：，]', '提出[^。；，撤]{0,5}请求[：，]',
    '诉请[：，]', '诉称[：，]', '请求[^。；，]{0,2}[：，]'
]
sucheng_pattern1 = '|'.join(sucheng_pattern1)


# sucheng_pattern2 = [
#     '原告[^。；未]*?提出[^。；，撤保]{0,5}申请', '提出[^。；，撤保]{0,5}申请[：，]',
# ]
# sucheng_pattern2 = '|'.join(sucheng_pattern2)


def sucheng_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('（[^。！？：<>]*?）', '', html_string)
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)

    if len(re.findall(introduction_pattern, html_string)) > 0:
        pattern = re.findall(introduction_pattern, html_string)[0]
        html_string = html_string[html_string.index(pattern) + len(pattern):]

    sucheng = []
    patterns = re.findall(sucheng_pattern1, html_string)
    if len(patterns) > 0:
        start_index = min([html_string.index(p) + len(p) for p in patterns])
        if html_string[start_index] in ['，', '：', '；']:
            start_index += 1
        html_string = html_string[start_index:]

        html_string = html_string.replace('</br>事实和理由：', '事实和理由：')
        if '</p>' in html_string:
            html_string = html_string[:html_string.index('</p>')]
        if '</br>' in html_string:
            html_string = html_string[:html_string.index('</br>')]

        if len(re.findall(biancheng_pattern, html_string)) > 0:
            pattern = re.findall(biancheng_pattern, html_string)[0]
            html_string = html_string[:html_string.index(pattern)]

        while len(html_string) > 0 and html_string[-1] not in ['。', '；', '，', '！', '？']:
            html_string = html_string[:-1]
        if len(html_string) == 0:
            return None
        sucheng.append(html_string)
        sucheng.append(suqing_extract(html_string))
    return sucheng if len(sucheng) > 0 else None


##############################################################################################################################################
#
# 提取被告辨称
#
##############################################################################################################################################

biancheng_pattern = '被告[^。；，：未]*辩称|辩称[：，]|被告[^。；，：]*承认|被告[^。；，：]*答辩认为|被告[^。；，：]*答辩意见：'


def biancheng_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('（[^。！？：<>]*?）', '', html_string)
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)

    if len(re.findall(introduction_pattern, html_string)) > 0:
        pattern = re.findall(introduction_pattern, html_string)[0]
        html_string = html_string[html_string.index(pattern) + len(pattern):]

    for pattern in renwei_pattern:
        if len(re.findall(pattern, html_string)) > 0:
            p = re.findall(pattern, html_string)[0][0]
            html_string = html_string[:html_string.rindex(p)]
            break

    html_string = html_string + '</p>'

    biancheng = []
    patterns = re.findall(biancheng_pattern, html_string)
    patterns += ['</p>']
    for i in range(len(patterns) - 1):
        start_index = html_string.index(patterns[i])
        if html_string[start_index] in ['。', '；', '，', '：']:
            start_index += 1
        html_string = html_string[start_index:]
        end_index = html_string.index(patterns[i + 1])
        infos = html_string[:end_index]
        if '</br>' in infos:
            infos = infos[:infos.index('</br>')]
        if len(infos) > 0:
            biancheng.append(infos)
    return biancheng if len(biancheng) > 0 else None


##############################################################################################################################################
#
# 提取证据
#
##############################################################################################################################################

proof_pattern = [
    '((认定|确认)(上述|以上|综上所述|前述).{0,3}(事实|实事)的证据(有|包括)[^。；]*[。；])',
    '((上述|以上|综上所述|前述).{0,3}(事实|实事)[^。;]*?(等.{0,2}证据|为据|为证|证实|佐证|为凭))',
    '((上述|以上|前述).{0,3}(事实|实事)[由有]下列证据[^。；]*?[；。])[^一二三四五六七八九十\d]',
    '((提供|提交|出具|举示|出示)[^。；]*等.{0,2}证据)',
    '((提供|提交|出具|举示|出示)[^。；]*证据(有|包括).*?[；。])[^一二三四五六七八九十\d]',
    '((提供|提交|出具|举示|出示)[^。；]*(以下|如下|下列)证据.*?[；。])[^一二三四五六七八九十\d]',
    '((证明|证实|佐证)[^。；，：]*证据(有|包括)[^。]*?。)',
    '[^未没]((提供|提交|出具|举示|出示)[^。；，：不未没无]*(证实))',
    '((提供|提交|出具|举示|出示)[^。；，：不未没无]*(为证据))',
    '((证据[一二三四五六七八九十\d][，；。、：][^，；。：]*?[，；。：]))',
]


def proof_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('[^。！？：]</br>', lambda x: x[0][0], html_string)

    html_string = html_string.replace('</p>', '').replace('<p>', '').replace('</br>', '')
    indices = []
    for pattern in proof_pattern:
        if len(re.findall(pattern, html_string)) > 0:
            for p in re.findall(pattern, html_string):
                index = html_string.index(p[0])
                indices.append((index, index + len(p[0])))
    i = 0
    while i < len(indices):
        j = i + 1
        while j < len(indices):
            if indices[i][0] > indices[j][1] or indices[i][1] < indices[j][0]:
                j += 1
                continue
            indices[i] = (min(indices[i][0], indices[j][0]), max(indices[i][1], indices[j][1]))
            indices.pop(j)
        i += 1

    proof = []
    filter_words = ['异议', '认为']
    for index in indices:
        flag = True
        for word in filter_words:
            if word in html_string[index[0]: index[1]]:
                flag = False
        if flag:
            proof.append(html_string[index[0]: index[1]])
    return proof if len(proof) > 0 else None


##############################################################################################################################################
#
# 提取查明事实
#
##############################################################################################################################################

chaming_pattern = [
    '(对(本案|案件).{0,3}事实.{0,2}(做|作)(如下|以下|下列)(归纳|认定|认证))',

    '((本案|案件)已查明.{0,3}事实(确认|确定|认定|认证)(如下|为))',
    '((本案|案件).{0,3}事实(确认|确定|证明|证实|查明|认定|查清|认证)(如下|为|是))',
    '((本案|案件).{0,3}事实(作|予以)(如下|以下|下列)(认定|认证))',

    '(对(本案|案件)(如下|以下|下列).{0,3}事实予以(确认|确定|证明|证实|查明|认定|查清|认证))',
    '(对(如下|以下|下列).{0,5}事实予以(确认|确定|证明|证实|查明|认定|查清|认证))',
    '(对(如下|以下|下列).{0,5}事实作(如下|以下|下列)(确认|确定|证明|证实|查明|认定|查清|认证))',

    '(本院.{0,5}(确认|确定|证明|证实|查明|认定|查清|认为|认证)(本案|案件)(如下|以下|下列).{0,3}事实)',
    '(本院.{0,5}(确认|确定|证明|证实|查明|认定|查清|认为|认证)(如下|以下|下列).{0,3}事实为本案.{0,3}事实)',
    '(本院.{0,5}(确认|确定|证明|证实|查明|认定|查清|认为|认证)(如下|以下|下列).{0,5}事实)',
    '(本院.{0,5}(确认|确定|证明|证实|查明|认定|查清|认为|认证).{0,5}事实(如下|为|是))',

    '(本院(查明|查清)(如下|以下|下列).{0,5}事实.{0,2}予以(确认|认定|认证))',

    '((确认|确定|证明|证实|查明|认定|查清|认为|认证)(本案|案件)(如下|以下|下列).{0,3}事实)',
    '((确认|确定|证明|证实|查明|认定|查清|认为|认证)(如下|以下|下列).{0,5}事实)',
    '((确认|确定|证明|证实|查明|认定|查清|认为|认证).{0,5}事实(如下|为|是))',

    '(经(审理查明|庭审查明|本院审查)[：，])',
    '((经审理|经审查|经庭审质证|经审理查明|经查明)(确认|确定|认定|认证)(如下|为))',
    '(经(本院审理|审理|庭审调查|开庭审理)(查明|确认|认定|认证)[：，])',
    '(本院.{0,5}(查明|查清)如下)',
    '((本案|案件).{0,3}事实(如下|为|是))',
    '(本院.{0,5}(查明|认定|查清)[：，])',
    '((<p>|</br>)(审理查明|查明|经查|经审理)[：，])',
]

end_sentences = ['((<p>本院认为))', '((</br>本院认为))', '((上述|以上|综上所述|前述).{0,3}(事实|实事))']


def pattern_extract(html_string, pattern):
    start_index = html_string.index(pattern) + len(pattern)
    while html_string[start_index] in [',', '，', ':', '：', ';', '；', '。', '的']:
        start_index += 1
    if html_string[start_index:start_index + 5] == '</br>':
        start_index += 5
    if html_string[start_index:start_index + 4] == '</p>':
        start_index += 4
    if html_string[start_index:start_index + 3] == '<p>':
        start_index += 3
    if '</p>' not in html_string[start_index:]:
        end_index = len(html_string) - 1
    else:
        end_index = start_index + html_string[start_index:].index('</p>')
    while html_string[end_index] not in [',', '，', ':', '：', ';', '；', '。']:
        end_index -= 1
    end_index += 1
    return html_string[start_index:end_index]


def chaming_fact_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)

    fact = None
    for p in chaming_pattern:
        if len(re.findall(p + '.{5,}', html_string)) > 0:
            pattern = re.findall(p, html_string)[0][0]
            fact = pattern_extract(html_string, pattern)
            for sentence in end_sentences:
                if len(re.findall(sentence, fact)) > 0:
                    pattern = re.findall(sentence, fact)[0][0]
                    fact = fact[:fact.rindex(pattern)]
            fact = fact.replace('</br>', '')
            if len(re.findall('与原告.{0,3}诉称[^，。；：！？不]*一致', fact)) > 0:
                fact = None
            break
    return fact


##############################################################################################################################################
#
# 提取争议焦点
#
##############################################################################################################################################


zhengyi_pattern = '争议.{0,3}焦点|焦点问题|调查重点'


def zhengyi_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)

    zhengyi = None
    if len(re.findall(zhengyi_pattern, html_string)) > 0:
        pattern = re.findall(zhengyi_pattern, html_string)[0]
        start_index = html_string.index(pattern)
        end_index = html_string.index(pattern)
        while html_string[start_index] not in ['，', ',', '。', '>', '？']:
            start_index -= 1
        start_index += 1
        while html_string[end_index] not in ['。', '？', '！']:
            end_index += 1
        zhengyi = html_string[start_index: end_index]
    return zhengyi


##############################################################################################################################################
#
# 提取本院认为
#
##############################################################################################################################################

renwei_pattern = [
    '(<p>本院(认为|.{0,1}审查认为))',
    '(</br>本院(认为|.{0,1}审查认为))',
    '(本院[^。；，？！]{0,5}(认为|.{0,1}审查认为))'
]


def renwei_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)
    not_found = True
    for pattern in renwei_pattern:
        if len(re.findall(pattern, html_string)) > 0:
            p = re.findall(pattern, html_string)[0][0]
            html_string = html_string[html_string.rindex(p):]
            not_found = False
            break
    if not_found:
        return None
    if html_string.startswith('<p>'):
        html_string = html_string[3:]
    if html_string.startswith('</br>'):
        html_string = html_string[5:]

    html_string = re.sub('：“.*?”', '，', html_string)
    html_string = re.sub('“.*?”', '', html_string)
    if len(re.findall(fatiao_pattern, html_string)) > 0:
        pattern = re.findall(fatiao_pattern, html_string)[-1][0]
        if len(re.findall(fatiao_pattern, pattern[1:])) > 0:
            pattern = re.findall(fatiao_pattern, pattern[1:])[-1][0]
        infos = html_string[:html_string.index(pattern)]
        for p in ['综上，', '鉴此，', '据此，', '为此，', '综上所述，', '故']:
            if infos.endswith(p):
                infos = infos[:-len(p)]
        if '</br>' in infos:
            return infos[:infos.index('</br>')]
        elif '</p>' in infos:
            return infos[:infos.index('</p>')]
        return infos
    elif '</p>' in html_string:
        return html_string[:html_string.index('</p>')]
    elif '</br>' in html_string:
        return html_string[:html_string.index('</br>')]
    else:
        return html_string


##############################################################################################################################################
#
# 提取本院认为中没有证据的事实描述
#
##############################################################################################################################################

no_proof_keywords = [
    '(无|没|缺乏|缺少).*证据',
    '(未|没).*(提交|提供|提出).*证据',
    '(未|没|难以|不足以).*(证明|证实|认定)',
    '证据不充分',
    '证据(不足以|不能).*(证明|证实)',
    '证据不足',
    '举不出.*证据',
    '举证不能'
]
no_proof_keywords = '(' + '|'.join('('+k+')' for k in no_proof_keywords) + ')'


def no_proof_extract(sentences, yuangao=None, beigao=None):
    if yuangao is not None:
        sentences = sentences.replace(yuangao, '原告')
    if beigao is not None:
        sentences = sentences.replace(beigao, '被告')
    sentences = sentences.replace('原告原告', '原告').replace('被告被告','被告')
    sentence_list = re.split('[。：；！？]', sentences)
    result = []
    for sentence in sentence_list:
        if '应当提供证据' in sentence:
            continue
        if '举证责任' in sentence:
            continue
        if '被告' in sentence and '原告' not in sentence:
            continue
        if '被告' in sentence and '原告' in sentence and sentence.index('被告') < sentence.index('原告'):
            continue
        for s in re.split('[，、（）]', sentence):
            if len(re.findall(no_proof_keywords, s))>0:
                pattern = re.findall(no_proof_keywords, s)[0][0]
                end_index = sentence.index(pattern)+len(pattern)
                while end_index<len(sentence) and sentence[end_index] not in ['，']:
                    end_index += 1
                sentence = sentence[:end_index]
                result.append(re.sub('，(因|且|但|由于)', '，', sentence))
                break
    return result


##############################################################################################################################################
#
# 提取法条
#
##############################################################################################################################################

fatiao_pattern = '((综上，|鉴此，|据此，|为此，|综上所述，|依照|根据|依据|按照)([^。：]*?)(判决|裁定|达成[^：；。]*?协议|判决（缺席）).{0,2}(：|；|。|，|</p>|</br>))'


def fatiao_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)
    html_string = re.sub('：“.*?”', '，', html_string)
    html_string = re.sub('“.*?”', '', html_string)

    fatiao = None
    if len(re.findall(fatiao_pattern, html_string)) > 0:
        html_string = re.findall(fatiao_pattern, html_string)[-1][0]
        if len(re.findall(fatiao_pattern, html_string[1:])) > 0:
            fatiao = re.findall(fatiao_pattern, html_string[1:])[-1][2]
        else:
            fatiao = re.findall(fatiao_pattern, html_string)[-1][2]
    return fatiao


##############################################################################################################################################
#
# 提取判决结果
#
##############################################################################################################################################

shouli_pattern = '案件受理费|案件诉讼费|本案受理费|本案诉讼费'
yanqi_pattern = '[，。；].{0,2}如.{0,2}未按.{0,2}判决'
shangsu_pattern = '[，。；].{0,2}如不服.{0,2}判决'


def panjue_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)

    for pattern in renwei_pattern:
        if len(re.findall(pattern, html_string)) > 0:
            p = re.findall(pattern, html_string)[0][0]
            html_string = html_string[html_string.rindex(p):]
            break

    html_string = re.sub('：“.*?”', '，', html_string)
    html_string = re.sub('“.*?”', '', html_string)
    if len(re.findall(fatiao_pattern, html_string)) > 0:
        pattern = re.findall(fatiao_pattern, html_string)[-1][0]
        if len(re.findall(fatiao_pattern, pattern[1:])) > 0:
            pattern = re.findall(fatiao_pattern, pattern[1:])[-1][0]
        html_string = html_string[html_string.rindex(pattern) + len(pattern):]
    elif len(re.findall('((判决如下|裁定如下|达成[^：；。]*?协议)(：|；|。|，|</p>|</br>))', html_string)) > 0:
        pattern = re.findall('((判决如下|裁定如下|达成[^：；。]*?协议)(：|；|。|，|</p>|</br>))', html_string)[0][0]
        html_string = html_string[html_string.index(pattern) + len(pattern):]
    elif '</p>' in html_string:
        html_string = html_string[html_string.index('</p>'):]

    for pattern in ['代审判员', '见习审判员', '代理审判员', '审判员', '代审判长', '见习审判长', '代理审判长', '审判长']:
        if pattern in html_string:
            html_string = html_string[:html_string.index(pattern)]
    infos = html_string.replace('</p>', '').replace('<p>', '').replace('</br>', '')

    panjue_result = {}
    patterns = re.findall(shouli_pattern + '|' + shangsu_pattern + '|' + yanqi_pattern, infos)
    end_index = min([infos.index(p) for p in patterns] + [len(infos)])
    panjue_result['判决'] = infos[:end_index] if end_index > 0 else None
    if len(re.findall(yanqi_pattern, infos)) > 0:
        pattern = re.findall(yanqi_pattern, infos)[0]
        start_index = infos.index(pattern) + 1
        patterns = re.findall(shouli_pattern + '|' + shangsu_pattern, infos[start_index:])
        end_index = min([infos.index(p) for p in patterns] + [len(infos)])
        panjue_result['延期'] = infos[start_index:end_index]
    if len(re.findall(shouli_pattern, infos)) > 0:
        pattern = re.findall(shouli_pattern, infos)[0]
        start_index = infos.index(pattern)
        patterns = re.findall(shangsu_pattern + '|' + yanqi_pattern, infos[start_index:])
        end_index = min([infos.index(p) for p in patterns] + [len(infos)])
        panjue_result['受理'] = infos[start_index:end_index]
    if len(re.findall(shangsu_pattern, infos)) > 0:
        pattern = re.findall(shangsu_pattern, infos)[0]
        start_index = infos.index(pattern) + 1
        patterns = re.findall(shouli_pattern + '|' + yanqi_pattern, infos[start_index:])
        end_index = min([infos.index(p) for p in patterns] + [len(infos)])
        panjue_result['上诉'] = infos[start_index:end_index]
    return panjue_result if len(panjue_result) > 0 else None


##############################################################################################################################################
#
# Y值打标
#
##############################################################################################################################################


def fact_suqing_split(suqing_sentences):
    """
    将陈述中的诉求和事实分离，按照特殊的词进行分离
    :param suqing_sentences: 陈述
    """
    suqing = suqing_sentences
    fact = suqing_sentences

    # 当陈述较短时，不分离诉求和事实
    if len(suqing_sentences) < 150 or (suqing_sentences.count('。') < 5 and len(suqing_sentences) < 200):
        if len(re.findall('事实.{0,1}理由：', suqing_sentences)) > 0:
            c = re.findall('事实.{0,1}理由：', suqing_sentences)
            start_new = suqing_sentences.rindex(c[0])
            suqing = suqing_sentences[:start_new]
            fact = suqing_sentences[start_new:]
        return suqing, fact

    pattern_list = ['请求.{0,6}判令', '请求.{0,6}判决', '要求.{0,6}判令', '申请.{0,6}判决',
                    '要求.{0,6}判决', '要求.{0,6}支付', '请求.{0,6}法院', '起诉.{0,6}请求',
                    '起诉.{0,6}要求', '要求.{0,6}判令', '诉.{0,6}判决', '诉[^讼]{0,6}请求',
                    '诉[^讼]{0,6}要求', '诉.{0,6}判令', '提.{0,6}诉请', '诉.{0,6}法院',
                    '提.{0,6}请求', '提.{0,6}诉讼', '诉[^讼]{0,6}本院', '现请求判令.{0,6}支付原告']
    key_signifier_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                          '一', '二', '三', '四', '五', '六', '七', '八', '九', ':', '：']
    split_sentence = re.findall(
        '(' + '：|'.join(pattern_list) + '：)', suqing_sentences)

    if len(split_sentence) == 0:
        split_sentence = re.findall(
            '(' + ':|'.join(pattern_list) + ':)', suqing_sentences)
    if len(split_sentence) == 0:
        split_sentence = re.findall(
            '(' + '|'.join(pattern_list) + ')', suqing_sentences)
    if len(split_sentence) == 0:
        split_sentence = re.findall('要求1|要求：|请求：|请法院.{0,6}判决：', suqing_sentences)
    if len(split_sentence) == 0:
        split_sentence = re.findall('原告要求|原告主张', suqing_sentences)
    if len(split_sentence) == 0:
        split_sentence = re.findall('(诉讼请求|原告诉称|要求|请求)', suqing_sentences)

    # 单独处理诉讼请求和请求依法判令
    split_sentence = [s for s in split_sentence if len(re.findall('[。；]', s)) == 0]
    if len(split_sentence) > 1:
        start = suqing_sentences.rindex(split_sentence[-1])
        if start + len(split_sentence[-1]) >= len(suqing_sentences) - 1:
            split_sentence.remove(split_sentence[-1])
    if len(split_sentence) > 1 and '原告诉称' in split_sentence:
        split_sentence.remove('原告诉称')
    if len(split_sentence) > 1 and '诉讼请求' in split_sentence:
        flag = suqing_sentences.rindex('诉讼请求')
        if flag + 4 < len(suqing_sentences) and suqing_sentences[flag + 4] not in key_signifier_list:
            split_sentence.remove('诉讼请求')
    if len(split_sentence) > 1 and '请求依法判决' in split_sentence:
        flag = suqing_sentences.rindex('请求依法判决')
        if flag + 5 < len(suqing_sentences) and suqing_sentences[flag + 5] not in key_signifier_list:
            split_sentence.remove('请求依法判决')

    start = 0
    end = 0
    if len(split_sentence) > 0:
        split_sentence_len = len(split_sentence)
        start = suqing_sentences.rindex(split_sentence[-1])
        fact = suqing_sentences[:start]
        add_len = len(split_sentence[-1])

        # 筛选关键词表，当关键词后含有特殊字符的优先选择
        if split_sentence_len > 1:
            for i in range(split_sentence_len - 1):
                if len(re.findall('(诉至|诉诸).*法院', split_sentence[i])) > 0:
                    continue
                start_flag = suqing_sentences.rindex(split_sentence[i])
                if start_flag + len(split_sentence[i]) + 1 > len(suqing_sentences) - 1:
                    break
                if suqing_sentences[start_flag + len(split_sentence[i])] in key_signifier_list + [',', '，']:
                    start = start_flag
                    add_len = len(split_sentence[i])
        end = start + add_len
        while start >= 0:
            if suqing_sentences[start] == "。":
                break
            start -= 1
        while end < len(suqing_sentences) - 1:
            if suqing_sentences[end] == "。":
                if end < len(suqing_sentences) - 1 and suqing_sentences[end + 1] not in key_signifier_list[:-2] + ['请']:
                    break
                elif end < len(suqing_sentences) - 2 and suqing_sentences[end + 1] in key_signifier_list[:-2] + [
                    '请'] and suqing_sentences[end + 2] in key_signifier_list[:9] + ['0']:
                    break
                else:
                    end += 1
            else:
                end += 1
        suqing = suqing_sentences[start + 1: end + 1]
        if split_sentence[-1] == '原告诉称':
            fact = suqing_sentences
        else:
            fact += suqing_sentences[end + 1:]

    # 构建干扰词表
    pattern_list_2 = ['提交.{0,6}证据', '支持.{0,6}诉讼',
                      '支持.{0,6}原告', '驳回.{0,6}原告', '如下举证', '提供.{0,6}证据', '请求.{0,6}变更', '证据.{0,4}：', '变更.{0,6}请求']
    split_sentence_2 = re.findall('(' + '|'.join(pattern_list_2) + ')', suqing)

    # 当存在干扰词或则提取诉请过短时，加上上一句话的信息
    if len(split_sentence_2) > 0:  # or len(suqing) < 50:
        if start > 0:
            start -= 1
        while start >= 0:
            if suqing_sentences[start] == '。':
                break
            start -= 1
        suqing = suqing_sentences[start + 1: end + 1]

    return suqing, fact


def fact_extract(suqing_sentences):
    """
    使用正则匹配过滤陈述中的非事实陈述
    :param suqing_sentences: 陈述语句
    :return: 过滤后的事实陈述
    """
    suqing_sentences = html_clean(suqing_sentences)
    suqing, fact = fact_suqing_split(suqing_sentences)
    is_fact_string = ""
    sentence_list = re.findall('.*?[^0-9][。；，;,:：]', fact + "。")
    for sentence in sentence_list:
        if sentence != "。" and len(sentence) > 5 and not sentence.startswith('如') and not sentence.startswith('若') and \
                len(re.findall("(要求|请求|诉求|诉称|诉请|起诉|认为|应当|应该|维护.*权益|如果|裁决|诉讼费|案.*受理费)", sentence)) == 0:
            is_fact_string += sentence

    return is_fact_string


# def suqing_extract(suqing_sentences):
#     """
#     返回提取的诉讼请求
#     :param suqing_sentences: 陈述语句
#     :return: 诉讼请求
#     """
#     suqing_sentences = html_clean(suqing_sentences)
#     suqing, fact = fact_suqing_split(suqing_sentences)
#     return suqing


def _sentence_keyword_match(row, sentence_list, func):
    """
    基于正负向关键词进行打标，如果有一个分句同时满足存在关键词并且不满足3个排除关键词，则返回1
    匹配到正向的了，还要看有没有负向的，没有负向的就直接返回1。我这边默认了1为空，23都为空
    :param row:
    :param sentence_list:
    :param func:
    :return:
    """
    for sentence in sentence_list:
        # 匹配正向关键词，匹配到则继续匹配负向关键词，匹配不到则跳到下一句
        if not func(row['positive_keywords'], sentence):
            continue
        # 检查负向关键词是否为空，空则返回1
        if row['negative_keywords1'] != row['negative_keywords1'] or row['negative_keywords1'] is None or len(
                row['negative_keywords1'].strip()) == 0:
            return 1
        # 匹配负向关键词，匹配到则继续匹配下一个负向关键词，匹配不到则跳到下一句
        if func(row['negative_keywords1'], sentence):
            continue
        if row['negative_keywords2'] != row['negative_keywords2'] or row['negative_keywords2'] is None or len(
                row['negative_keywords2'].strip()) == 0:
            return 1
        if func(row['negative_keywords2'], sentence):
            continue
        if row['negative_keywords3'] != row['negative_keywords3'] or row['negative_keywords3'] is None or len(
                row['negative_keywords3'].strip()) == 0:
            return 1
        if not func(row['negative_keywords3'], sentence):
            return 1
    return 0


def get_panjue_label_from_config(panjue_sentences, anyou, suqing_sentences, suqiu_anyou_list, panjue_config,
                                 problem='劳动纠纷', labor_label_mode=True):
    """
    根据诉请和事实理由，解析出诉求类型
    :param panjue_sentences: 判决内容
    :param anyou: 案由
    :param suqing_sentences: 诉求内容
    :return:
    """
    # 1.筛选诉求
    suqius = suqiu_anyou_list[problem][suqiu_anyou_list[problem].apply(lambda x: True if anyou in x else False)]

    # 2.检测每个诉求是否支持
    panjue_sentence_list = re.split('[。；]', panjue_sentences)  # todo 目前转为使用短句匹配，因为长度匹配问题太多。

    # 获取案由对应的诉求关键词信息
    temp = panjue_config[(panjue_config['apply_to_panjue'] == 1) & (panjue_config['problem'] == problem) & (
        panjue_config['suqiu'].isin(suqius.index))].copy()
    # 该案由没有对应诉求则返回空
    if len(temp) == 0:
        return {}
    # 每个诉求对判决打标
    temp['panjue_label'] = temp.apply(_sentence_keyword_match, axis=1,
                                      args=(panjue_sentence_list, check_pattern_with_panjue_sentence,))
    panjue_label = temp['panjue_label'].groupby(temp['suqiu']).agg(lambda x: max(x))
    panjue_label = {r: panjue_label[r] for r in panjue_label.index}
    # print(panjue_label)

    # # 3.1 添加确定性例外处理：如果案由是'确认劳动关系纠纷‘，优先使用如下规则
    # if problem == '劳动纠纷' and anyou == '确认劳动关系纠纷':
    #     if '驳回' not in panjue_sentence_list[0] and '不予支持' not in panjue_sentence_list[0] \
    #             and len(re.findall(negative_word, panjue_sentence_list[0])) == 0 \
    #             and len(re.findall('解除', panjue_sentence_list[0])) == 0:
    #         panjue_label['存在劳动关系'] = 1

    # 3.2 添加确定性例外处理：如果判决是'付xxx元'，那么'支付工资'=1
    if problem == '劳动纠纷':
        for result_string_ in panjue_sentence_list:
            if not result_string_.startswith("案件受理费10元"):
                flag1 = len(re.findall('付.*元', result_string_)) > 0
                flag2 = len(re.findall('驳回.*付.*元', result_string_)) > 0
                flag3 = len(re.findall('付.*元.*不予支持', result_string_)) > 0
                flag4 = len(re.findall('付.*(补偿|赔偿|加班|工伤|鉴定|养老补助|材料费|保健食品费|医疗费|代通金|垫付款|仲裁费).*元', result_string_)) > 0
                if flag1 and not flag2 and not flag3 and not flag4:
                    panjue_label['支付工资'] = 1
                    break

    # # 3.3 如果支持工资或xxx金为1，那么将存在劳动关系，标记为1.
    # if problem == '劳动纠纷' and labor_label_mode:
    #     for k, v in panjue_label.items():
    #         if v == 1 and '存在劳动关系' in panjue_label:
    #             panjue_label['存在劳动关系'] = 1

    return panjue_label


def get_suqiu_label_from_config(panjue_sentences, anyou, suqing_sentences, suqiu_anyou_list, panjue_config,
                                problem='劳动纠纷'):
    """
    根据诉请和事实理由，解析出诉求类型.
    默认suqiu_anyou_list可以从
    :param panjue_sentences: 判决
    :param anyou: 案由
    :param suqing_sentences: 诉请和事实理由
    :return:
    """
    # 提取诉讼请求
    suqing_sentence_list = re.split('[。；]', suqing_sentences)
    # suqing_sentence_list = re.split('[。；]', suqing_extract(suqing_sentences))

    # 1.筛选诉求
    # print("###suqius:",suqius)
    suqius = suqiu_anyou_list[problem][suqiu_anyou_list[problem].apply(lambda x: True if anyou in x else False)]

    # 2.检测每个诉求
    # 获取案由对应的诉求关键词信息
    # print("panjue_config_v:",panjue_config_v)
    temp = panjue_config[(panjue_config['apply_to_suqiu'] == 1) & (panjue_config['problem'] == problem) & (
        panjue_config['suqiu'].isin(suqius.index))].copy()
    # 该案由没有对应诉求则返回空
    if len(temp) == 0:
        print("#######################案由没有对应的诉求，返回空，请检查.################################################")
        return {}

    temp['suqiu_label'] = temp.apply(_sentence_keyword_match, axis=1,
                                     args=(suqing_sentence_list, check_pattern_with_suqing_sentence,))
    suqiu_label = temp['suqiu_label'].groupby(temp['suqiu']).agg(lambda x: max(x))
    suqiu_label = {r: suqiu_label[r] for r in suqiu_label.index}
    # print(suqiu_label)

    # 3.1 添加确定性例外处理：工资和加班工资同时存在
    # if problem == '劳动纠纷':
    #     for su_qing_ in suqing_sentence_list:
    #         if len(re.findall('工资', su_qing_)) > len(re.findall('加班工资', su_qing_)) \
    #                 and not check_pattern_with_suqing_sentence('以.*工资.*基数|补偿|替代提前通知期|代通金|赔偿|竞业限制|双倍工资|二倍工资', su_qing_):
    #             suqiu_label['支付工资'] = 1

    # print("###suqiu_label:",suqiu_label)
    return suqiu_label


def check_pattern_with_panjue_sentence(pattern, panjue_sentence, filter_word=None):
    """
    使用句式匹配判决：遍历每一个分句，如果任何一个分句有正向匹配，不存在负向匹配且没有驳回，那么认为匹配到了；否则，没有匹配到
    :param pattern:
    :param panjue_sentence:
    :param filter_word: 过滤词
    :return:
    """
    # pattern2 = pattern.replace('.+?','')
    pattern3 = '，|。|；|：'
    # panjuan0 = re.findall(pattern,panjue_sentence)
    # pattern=pattern.replace('.+?','')
    panjuan0 = [x.group() for x in re.finditer(pattern, panjue_sentence)]
    # print('1111',panjuan0)
    panjuan = []
    if len(panjuan0) > 0:
        for panjue_i in panjuan0:
            temp = panjue_i
            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            panjuan.append(temp)

            temp = panjue_i
            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            panjuan.append(temp)

    # print(panjuan)
    flag_pos_1 = len(panjuan) > 0
    # flag_pos_2 = len(re.findall(pattern2, panjue_sentence)) > 0
    flag_pos = flag_pos_1  # or flag_pos_2

    # flag_neg1 = (len(re.findall(negative_word + '.*' + pattern, result)) > 0 and '未休' not in result and '无锡' not in result)
    # flag_neg2 = (len(re.findall(pattern + '.*' + negative_word, result)) > 0 and '未休' not in result and '无锡' not in result)
    flag_neg3 = len(re.findall('驳回.*' + pattern, panjue_sentence)) > 0
    flag_neg4 = len(re.findall(pattern + '.*不予', panjue_sentence)) > 0
    flag_neg5 = len(re.findall('不准.*' + pattern, panjue_sentence)) > 0
    if flag_pos and not flag_neg3 and not flag_neg4 and not flag_neg5 and not (
            filter_word is not None and filter_word in panjue_sentence):
        return True
    return False


def check_pattern_with_suqing_sentence(pattern, suqing_sentence, filter_word=None):
    """
    使用句式匹配诉求：
    :param pattern:
    :param suqing_sentence: 诉求和事实
    :param filter_word: 过滤词
    :return:
    """
    # pattern2 = pattern.replace('.+?','')
    pattern3 = '，|。|；|：'
    # pattern4 = pattern.replace('.+?','.*')
    # print("pattern:",pattern,";suqing_sentence:",suqing_sentence)
    panjuan0 = [x.group() for x in re.finditer(pattern, suqing_sentence)]
    # print(panjuan0)
    panjuan = []
    if len(panjuan0) > 0:
        for panjue_i in panjuan0:
            temp = panjue_i
            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            panjuan.append(temp)

            temp = panjue_i
            while len(re.findall(pattern, temp[1:])) > 0:
                new_pipei = re.search(pattern, temp[1:])
                temp = new_pipei.group()

            while len(re.findall(pattern, temp[:-1])) > 0:
                new_pipei = re.search(pattern, temp[:-1])
                temp = new_pipei.group()

            panjuan.append(temp)

    # print(panjuan)
    flag_pos_1 = len(panjuan) > 0
    # flag_pos_2 = len(re.findall(pattern2, suqing_sentence)) > 0
    flag_pos = flag_pos_1  # or flag_pos_2
    # print(flag_pos)

    if flag_pos and not (filter_word is not None and filter_word in suqing_sentence):
        # print("suqing:",suqing,";flag_pos:",flag_pos,";pattern:",pattern)
        return True
    return False


def get_panjue_label_filter_by_suqiu(panjue_sentences, anyou, suqing_sentences, problem='劳动纠纷'):
    """
    根据判决结果，结合诉求提取出标签。是一个多标签，每个诉求对应一个标签(0,1)；
    '存在劳动关系'，将被标记为1，如果诉讼请求中有'存在劳动关系'类型的请求，并且判决中判了劳动关系。其他诉求也类似处理。
    :param panjue_sentences:
    :param anyou:
    :param suqing_sentences:
    :return:
    """
    panjue_config = get_panjue_config()
    suqiu_anyou_list = panjue_config['anyou'].groupby([panjue_config['problem'], panjue_config['suqiu']]).agg(
        lambda x: list(x)[0])
    suqiu_anyou_list = suqiu_anyou_list.str.split('|')

    # 1.看诉讼请求
    suqiu_label = get_suqiu_label_from_config(panjue_sentences, anyou, suqing_sentences, suqiu_anyou_list,
                                              panjue_config, problem=problem)  # ADD by xul 2018-12-05
    print("1.suqiu_label:", suqiu_label)
    # 2.看判决
    panjue_label = get_panjue_label_from_config(panjue_sentences, anyou, suqing_sentences, suqiu_anyou_list,
                                                panjue_config, problem=problem)  # ADD by xul 2018-12-05
    print("2.panjue_label:", panjue_label)

    # 3.组织结果
    panjue_label_filtered = {}
    for suqiu, label in panjue_label.items():
        if suqiu in suqiu_label and suqiu_label[suqiu] == 1:
            bbb = int(label)
            panjue_label_filtered[suqiu] = bbb
        else:
            panjue_label_filtered[suqiu] = -1
    return panjue_label_filtered


def get_label_string(row, suqiu_anyou_list, panjue_config):
    """
    对DataFrame每行数据进行打标
    :param row: DataFrame行
    :return: 打标结果，字符串
    """
    # 诉求打标
    if row['suqing']!=row['suqing'] or row['suqing'] is None:
        return ''

    suqiu_labels = get_suqiu_label_from_config(
        row['panjue'], row['anyou'], row['suqing'], suqiu_anyou_list, panjue_config, problem=row['problem'])
    # 判决打标
    panjue_labels = get_panjue_label_from_config(
        row['panjue'], row['anyou'], row['suqing'], suqiu_anyou_list, panjue_config, problem=row['problem'])

    result = []
    for suqiu, value in panjue_labels.items():
        if suqiu in suqiu_labels and suqiu_labels[suqiu] == 1:
            result.append(suqiu + ':' + str(int(value)))
        else:
            result.append(suqiu + ':-1')
    return ';'.join(result)


def get_problem_suqiu_from_config(anyou, suqing_sentences, suqiu_anyou_list, panjue_config):
    """
    根据诉请和事实理由，解析出诉求类型.
    默认suqiu_anyou_list可以从
    :param panjue_sentences: 判决
    :param anyou: 案由
    :param suqing_sentences: 诉请和事实理由
    :return:
    """
    # 1.筛选纠纷类型
    problems = panjue_config[panjue_config['anyou'].apply(lambda x: True if anyou in x.split('|') else False)][
        'problem']

    # 2.检测每个纠纷类型
    result = {}
    for problem in problems:
        suqiu_label = get_suqiu_label_from_config(None, anyou, suqing_sentences, suqiu_anyou_list, panjue_config,
                                                  problem)
        suqius = [suqiu for suqiu, value in suqiu_label.items() if value == 1]
        result[problem] = suqius
    return result


def get_panjue_config(from_database=True):
    if from_database:
        connect = pymysql.connect(host=host_ip,
                                  user="justice_user_03",
                                  password="justice_user_03_pd_!@#$",
                                  db="justice")
        sql = '''
             select problem, suqiu, anyou, logic, positive_keywords, negative_keywords1, negative_keywords2, negative_keywords3,
             apply_to_panjue, apply_to_suqiu, suqiu_desc
             from algo_train_law_case_y_keyword
             where (status = 1 and problem not in ('%s')) or (status = %s and problem in ('%s'))
        ''' % ("','".join(new_problems), table_status, "','".join(new_problems))
        panjue_config = pd.read_sql(sql, con=connect)
        # panjue_config['positive_keywords'] = panjue_config['positive_keywords'].apply(keyword_list_expand)
        # panjue_config['negative_keywords1'] = panjue_config['negative_keywords1'].apply(keyword_list_expand)
        # panjue_config['negative_keywords2'] = panjue_config['negative_keywords2'].apply(keyword_list_expand)
        # panjue_config['negative_keywords3'] = panjue_config['negative_keywords3'].apply(keyword_list_expand)
        panjue_config.to_csv(config_path + 'panjue.csv')
        connect.close()
    else:
        panjue_config = pd.read_csv(config_path + 'panjue.csv')
    return panjue_config


##############################################################################################################################################
#
# 特征关键词配置文件导出
#
##############################################################################################################################################


def _get_problem_suqiu_factor_keyword():
    """
    根据大的案由名称（如，劳动纠纷）的ID，得到下面的pattern列表；以及一个字典：可以根据pattern，得到对应的keyword、benefit_type和benefit_type
    :return: pattern_list,pattern_keyword_benefittype_dict

    """

    def _keyword_correct(keyword):
        keyword = keyword.rstrip(".*").rstrip(".").lstrip(".*").lstrip(".")
        keyword = keyword.replace("（", "(").replace("）", ")").replace("**", ".*").replace(',*', '.*')
        if ")" in keyword and "(" not in keyword:
            keyword = keyword.replace(")", "")
        keyword = re.sub(r'[\u4E00-\u9FA5]{1}\*', lambda x: x.group(0)[0] + '.' + x.group(0)[1], keyword)
        return keyword

    # 1. 连接数据库
    sql = '''
        select distinct k.name, k.pattern keyword, k.benefit_type, f.id factor_id, f.factor_name, a.appeal suqiu, a.reason_name problem
        from algo_train_law_case_reason_factor f join algo_train_law_case_reason_extrainfo a join algo_train_law_factor_keyword k
        on f.appeal_id=a.id and k.factor_id=f.id 
        where (a.status=1 and k.status=1 and f.status=1 and a.reason_name not in ('%s')) 
        or (a.status=%s and k.status=%s and f.status=%s and a.reason_name in ('%s'))
    ''' % ("','".join(new_problems), table_status, table_status, table_status, "','".join(new_problems))
    connect = pymysql.connect(host=host_ip,
                              user="justice_user_03",
                              password="justice_user_03_pd_!@#$",
                              db="justice")
    df = pd.read_sql(sql, con=connect)

    # 2.将pattern和对应的keyword,benefit_type放入dict
    df['keyword'] = df['keyword'].apply(_keyword_correct)
    df = df.sort_values(by=['problem', 'suqiu', 'factor_name', 'keyword'])
    # problem_keyword_benefit_dict = df['benefit_dict'].groupby([df['problem'], df['keyword']]).agg(lambda x: '$'.join(sorted(set(x))))
    #
    # problem_suqiu_keyword_benefit_dict = df['benefit_dict'].groupby([df['problem'], df['suqiu'], df['keyword']]).agg(lambda x: '$'.join(sorted(set(x))))
    #
    # factorid_name_dict = df['factor_name'].groupby(df['factor_id']).agg(lambda x: list(x)[0])

    # problem_factor_keyword_dict = df['keyword'].groupby([df['problem'], df['factor_name']]).agg(
    #     lambda x: '#'.join(sorted(set(x))))
    #
    # keyword_suqiu_factor_dict = df['factor_name'].groupby(df['keyword_suqiu']).agg(lambda x: list(x)[0])
    #
    # problem_suqiu_keyword_dict = df['keyword'].groupby([df['problem'], df['suqiu']]).agg(
    #     lambda x: '#'.join(sorted(set(x))))
    # problem_suqiu_keyword_benefit_dict = df['benefit_dict'].groupby([df['problem'], df['suqiu'], df['keyword']]).agg(
    #     lambda x: list(x)[0])
    #
    # problem_suqiu_factor_dict = df['factor_name'].groupby([df['problem'], df['suqiu']]).agg(
    #     lambda x: '#'.join(sorted(set(x))))
    # problem_suqiu_factor_keyword_dict = df['keyword'].groupby([df['problem'], df['suqiu'], df['factor_name']]).agg(
    #     lambda x: '#'.join(sorted(set(x))))
    # problem_suqiu_factor_benefit_dict = df['benefit_type'].groupby([df['problem'], df['suqiu'], df['factor_name']]).agg(
    #     lambda x: list(x)[0])

    # 3. 关闭数据库连接
    connect.close()
    return df[['problem', 'suqiu', 'factor_id', 'factor_name', 'keyword', 'benefit_type', 'name']]


def initial_problem_suqiu_factor_keyword():
    """
    初始化诉求特征数据
    """
    df = _get_problem_suqiu_factor_keyword()

    if not os.path.exists(config_path):
        os.mkdir(config_path)

    df.to_csv(config_path + 'problem_suqiu_factor_keyword_benefit_dict.csv', index=False, encoding='utf-8')


def get_problem_suqiu_factor_keyword_benefit_dict():
    if not os.path.exists(config_path + 'problem_suqiu_factor_keyword_benefit_dict.csv'):
        initial_problem_suqiu_factor_keyword()
    problem_suqiu_factor_keyword_dict = pd.read_csv(config_path + 'problem_suqiu_factor_keyword_benefit_dict.csv',
                                                    encoding='utf-8')
    return problem_suqiu_factor_keyword_dict


def get_all_keywords():
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    all_keywords = sorted(df['keyword'].drop_duplicates())
    return all_keywords


def get_problem_keyword_dict():
    """
    从文件读取问题类型的关键词列表，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_keyword_dict = df['keyword'].groupby(df['problem']).agg(lambda x: sorted(set(x)))
    return problem_keyword_dict


def get_problem_keyword_factor_dict():
    """
    从文件读取关键词对应的特征，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_keyword_factor_dict = df['factor_name'].groupby([df['problem'], df['keyword']]).agg(
        lambda x: sorted(set(x)))
    return problem_keyword_factor_dict


def get_all_factors():
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    all_factors = sorted(df['factor_name'].drop_duplicates())
    return all_factors


def get_problem_factors():
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_factors = df['factor_name'].groupby(df['problem']).agg(lambda x: sorted(set(x)))
    return problem_factors


def get_problem_suqiu_keyword_dict():
    """
    从文件读取问题类型的诉求对应的关键词列表，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_suqiu_keyword_dict = df['keyword'].groupby([df['problem'], df['suqiu']]).agg(lambda x: sorted(set(x)))
    return problem_suqiu_keyword_dict


def get_problem_suqiu_keyword_factor_dict():
    """
    从文件读取问题类型的诉求对应的关键词列表，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_suqiu_keyword_dict = df['factor_name'].groupby([df['problem'], df['suqiu'], df['keyword']]).agg(
        lambda x: sorted(set(x)))
    return problem_suqiu_keyword_dict


def get_problem_suqiu_keyword_benefit_dict():
    """
    从文件读取关键词对应的有利不利状态，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    df['benefit'] = df.apply(lambda row: row['name'] + '#' + row['benefit_type'] + '#' + row['factor_name'], axis=1)
    problem_suqiu_keyword_benefit_dict = df['benefit'].groupby([df['problem'], df['suqiu'], df['keyword']]).agg(
        lambda x: sorted(set(x)))
    problem_suqiu_keyword_benefit_dict = problem_suqiu_keyword_benefit_dict.apply(lambda x: [t.split('#') for t in x])
    return problem_suqiu_keyword_benefit_dict


def get_problem_suqiu_factor_dict():
    """
    从文件读取问题类型对应的特征列表，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_suqiu_factor_dict = df['factor_name'].groupby([df['problem'], df['suqiu']]).agg(lambda x: sorted(set(x)))
    return problem_suqiu_factor_dict


def get_problem_suqiu_factorid_name_dict():
    """
    从文件读取特征id和特征对应关系，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_suqiu_factorid_name_dict = df['factor_name'].groupby([df['problem'], df['suqiu'], df['factor_id']]).agg(
        lambda x: list(x)[0])
    return problem_suqiu_factorid_name_dict


def get_problem_suqiu_factor_keyword_dict():
    """
    从文件读取特征对应的关键词列表，如文件不存在则从数据库读取
    :return:
    """
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_suqiu_factor_keyword_dict = df['keyword'].groupby([df['problem'], df['suqiu'], df['factor_name']]).agg(
        lambda x: sorted(set(x)))
    return problem_suqiu_factor_keyword_dict


def get_problem_suqiu_factor_benefit_dict():
    df = get_problem_suqiu_factor_keyword_benefit_dict()
    problem_suqiu_factor_benefit_dict = df['benefit_type'].groupby([df['problem'], df['suqiu'], df['factor_name']]).agg(
        lambda x: sorted(set(x)))
    return problem_suqiu_factor_benefit_dict


##############################################################################################################################################
#
# 问答数据
#
##############################################################################################################################################

def get_suqiu_qaid_dict():
    # 1. 连接数据库
    connect = pymysql.connect(host=host_ip,
                              user="justice_user_03",
                              password="justice_user_03_pd_!@#$",
                              db="justice")

    # 2. 查询:诉求与案由对应关系表
    question_sql = '''
        select q.appeal_id, q.factor_id
        from algo_train_law_case_reason_qnr q join algo_train_law_case_reason_extrainfo a 
        on q.appeal_id = a.id
        where q.factor_id is not null and ((a.status=1 and a.reason_name not in ('%s')) or (a.status=%s and a.reason_name in ('%s')))
    ''' % ("','".join(new_problems), table_status, "','".join(new_problems))
    question_table = pd.read_sql(question_sql, con=connect)
    suqiu_qaid_dict = question_table['factor_id'].groupby(question_table['appeal_id']).agg(lambda x: sorted(set(x)))

    connect.close()
    return suqiu_qaid_dict


def get_problem_suqiu_id_dict():
    # 1. 连接数据库
    db_based_law = pymysql.connect(host=host_ip,
                                   user="justice_user_03",
                                   password="justice_user_03_pd_!@#$",
                                   db="justice")

    # 2. 查询:诉求与案由对应关系表
    suqiu_sql = '''
        select id, appeal as suqiu, reason_name as problem 
        from algo_train_law_case_reason_extrainfo 
        where (status=1 and reason_name not in ('%s')) or (status=%s and reason_name in ('%s'))
    ''' % ("','".join(new_problems), table_status, "','".join(new_problems))
    suqiu_table = pd.read_sql(suqiu_sql, con=db_based_law)
    problem_suqiu_id_dict = suqiu_table['id'].groupby([suqiu_table['problem'], suqiu_table['suqiu']]).agg(
        lambda x: list(x)[0])

    db_based_law.close()
    return problem_suqiu_id_dict


def get_problem_suqiu_qa_factor_dict():
    # 1. 连接数据库
    db_based_law = pymysql.connect(host=host_ip,
                                   user="justice_user_03",
                                   password="justice_user_03_pd_!@#$",
                                   db="justice")

    # 2. 查询:诉求与案由对应关系表
    question_sql = '''
        select a.reason_name problem, a.appeal suqiu, q.question_summary factor, q.question 
        from algo_train_law_case_reason_qnr q left join algo_train_law_case_reason_extrainfo a
        on q.appeal_id = a.id 
        where (a.status=1 and q.status=1 and reason_name not in ('%s')) 
        or (a.status=%s and q.status=%s and reason_name in ('%s'))
    ''' % ("','".join(new_problems), table_status, table_status, "','".join(new_problems))
    question_table = pd.read_sql(question_sql, con=db_based_law)
    problem_suqiu_qa_factor_dict = question_table['factor'].groupby(
        [question_table['problem'], question_table['suqiu'], question_table['question']]).agg(lambda x: list(x)[0])

    db_based_law.close()
    return problem_suqiu_qa_factor_dict


##############################################################################################################################################
#
# X特征匹配
#
##############################################################################################################################################

# 特征里没有否定意义的关键词
no_negative_factor = {
    '财产关系': '.*',
    '存在借贷关系': '(合同|协议)',
    '存在租赁关系': '(合同|协议|租赁)',
    '存在劳务关系': '(合同|协议)',
    '劳动关系': '(合同|协议)',
}

# 没有否定意义的关键词
no_negative_keyword = ['公司', '单位', '工厂', '夫妻', '夫妻共同财产', '患病']
no_negative_keyword = keyword_list_expand(no_negative_keyword)

# 关键词没有正向意义的情形
no_positive_pattern = {
    '向.*借': '向.*借(款|据|条)',
    '逾期还款': '逾期还款(达.*天|应)',
    '在.*工作': '存在.*工作',
    '居间合同': '口头居间合同',
}
no_positive_pattern = {keyword_list_expand(k): v for k, v in no_positive_pattern.items()}

# 关键词没有否定意义的情形
no_negative_pattern = [
    ('欠', '欠.*(未|没|不).*(发|付)'),
    ('务工', '务工.*未领取'),
    ('固定期限劳动合同', '无固定期限劳动合同'),
    ('加班', '加班不给工资'),
    ('离婚', '感情不和.*离婚'),
    ('离婚', '离婚.*未分割.*财产')
]

# 没有证据特征关键词
renwei_extra_keywords = [
    '(无|没|缺乏|缺少).*(事实|法律).*(依据|根据|证据)', '(无|没|缺乏|缺少).*(依据|根据|证据)',
    '(未|没).*(提交|提供).*证据', '(未|没|难以).*(证明|证实)',
    '证据不充分', '(无|没|没有|缺乏|缺少)依据', '于法无据', '(无|没|没有|缺乏|缺少)相应依据', '证据(不足以|不能).*(证明|证实)',
    '(理据|证据|依据)不足', '举不出.*证据', '举证不能'
]


def get_data_factor_features(data, keyword_list, pre_name_keyword='keyword:'):
    """
    获取特征匹配数据
    :param row:
    :param columns:
    :return:
    """

    def _match(row, cs):
        result = min([row[c] for c in cs])
        if result == 0:
            result = max([row[c] for c in cs])
        return result

    if len(data) == 0:
        raise ValueError('data size is 0')
    else:
        columns = []
        for keyword in keyword_list:
            if (pre_name_keyword + keyword) in data.columns:
                columns.append(pre_name_keyword + keyword)
        if len(columns) == 0:
            return np.zeros(len(data), dtype=int)
        else:
            return data.apply(_match, axis=1, args=(columns,)).values


def get_factor_matching_keywords(data, factor_keywords, pre_name_keyword='keyword:', pre_name_factor='factor:'):
    """
    获取特征匹配结果
    :param data: DataFrame
    :param factor_keywords: 特征关键词表
    :return:
    """
    matched_factors = []
    notmatched_factors = []
    for index, row in data.iterrows():
        mf = []
        nf = []
        for factor, keyword_list in factor_keywords[row["problem"]][row['suqiu']].items():
            if pre_name_factor + factor in row:
                keywords = []
                for keyword in keyword_list:
                    if pre_name_keyword + keyword in row and row[pre_name_keyword + keyword] in [1, -1]:
                        keywords.append('%s(%s)' % (keyword, row[pre_name_keyword + keyword]))
                if row[pre_name_factor + factor] == 1:
                    mf.append(factor + ':' + ';'.join(keywords))
                else:
                    nf.append(factor + ':' + ';'.join(keywords))
            else:
                nf.append(factor + ":")
        matched_factors.append('\n'.join(mf))
        notmatched_factors.append('\n'.join(nf))
    return matched_factors, notmatched_factors


def get_matching_keywords(data, suqiu_keywords, pre_name_keyword):
    """

    :param data:
    :param factor_keywords:
    :return:
    """
    matched_keywords = []
    for index, row in data.iterrows():
        mk = []
        for keyword in suqiu_keywords[row['problem']][row['suqiu']]:
            if pre_name_keyword + keyword in row and row[pre_name_keyword + keyword] in [1, -1]:
                mk.append(keyword)
        matched_keywords.append('\n'.join(mk))
    return matched_keywords


def multi_processing_data(lines, process_num, keyword_list, keyword_factor, use_tongyici=True):
    """

    :param process_num:
    :param problem:
    :param num_examples:
    :param num_training:
    :param num_validation:
    :return:
    """
    if use_tongyici:
        keyword_list_expanded = keyword_list_expand(keyword_list)
        keyword_factor_expanded = {keyword_list_expand(k): v for k, v in keyword_factor.items()}
    else:
        keyword_list_expanded = keyword_list
        keyword_factor_expanded = keyword_factor

    print("multiprocessing of extract feature.started;process_num:", process_num)
    chunks = build_chunk(lines, chunk_num=process_num - 1)  # 4.1 split data as chunks
    pool = multiprocessing.Pool(processes=process_num)
    for chunk_id, each_chunk in enumerate(chunks):  # 4.2 process each chunk,and save result to file system
        pool.apply_async(get_X, args=(
            each_chunk, keyword_list_expanded, keyword_factor_expanded, "tmp_" + str(chunk_id)))  # apply_async
    pool.close()
    pool.join()

    print("allocate work load finished. start map stage.")
    X = np.zeros((len(lines), len(keyword_list)))  # 2 represent special features added manual
    index = 0
    for chunk_id in range(process_num):  # 4.3 merge sub file to final file.
        temp_file_name = "tmp_" + str(chunk_id) + ".npy"  # get file name
        x_temp = np.load(
            temp_file_name)  # load file FileNotFoundError: [Errno 2] No such file or directory: 'inference_with_reason/data/input_x_0'
        num_example_temp, feature_size = x_temp.shape  # get shape of data
        X[index:index + num_example_temp] = x_temp  # assign a sub array to big array
        index += num_example_temp  # increment index
        rm_command = 'rm ' + temp_file_name
        os.system(rm_command)

    print("multiprocessing of extract feature.ended.")
    return X


def build_chunk(lines, chunk_num=4):
    """
    :param lines: total thing
    :param chunk_num: num of chunks
    :return: return chunks but the last chunk may not be equal to chunk_size
    """
    total = len(lines)
    chunk_size = float(total) / float(chunk_num + 1)
    chunks = []
    for i in range(chunk_num + 1):
        if i == chunk_num:
            chunks.append(lines[int(i * chunk_size):])
        else:
            chunks.append(lines[int(i * chunk_size):int((i + 1) * chunk_size)])
    return chunks


def get_X(lines_training, keyword_list, keyword_factor, target_file):
    """
    匹配输入，得到特征
    :param lines_training:
    :param pattern_list:
    :return:
    """
    try:
        X = []
        # print("length of lines(traing/valid/test):", len(lines_training))
        for i, line in enumerate(lines_training):
            feature_list, _ = single_case_match(line, keyword_list, keyword_factor)
            X.append(feature_list)
        np.save(target_file, X)
    except BaseException:
        traceback.print_exc()


# def single_case_match(input, keyword_list, keyword_factor, yuangao_type, shuqing_type, task_type):  # todo
#     """
#     匹配单独的一个输入(多个句子)，得到特征
#     :param input:
#     :param problem:
#     :return:
#     """
#     feature_list = np.zeros(len(keyword_list) + 2, dtype=int)  # 初始化特征序列
#
#     input = remove_law_from_input(input)  # 去掉法条
#     if task_type in ['fact', 'chaming']:
#         sentence_list = re.split('[。；，：,;:]', input)
#     else:
#         sentence_list = []
#         for s in re.split('[。；;：:]', input):
#             if '被告' not in s or ('被告' in s and '原告' in s and s.index('被告') > s.index('原告')):
#                 sentence_list += re.split('[，,]', s)
#     for k, key_word in enumerate(keyword_list):  # 针对每一个关键词，设法去匹配
#         if len(re.findall(key_word.replace('.*', '[^。；，：,;:]*'), input)) == 0:
#             continue
#         if task_type in ['fact', 'chaming']:
#             factor = keyword_factor[key_word][0]
#             for index, sentence in enumerate(sentence_list):  # 针对每个句子去匹配
#                 flag = match_sentence_by_keyword(sentence, factor, key_word)
#                 if flag == 1 or flag == -1:
#                     feature_list[k] = flag
#                     break
#         else:
#             for index, sentence in enumerate(sentence_list):  # 针对每个句子去匹配
#                 if len(re.findall(key_word, sentence)) > 0:
#                     feature_list[k] = 1
#                     break
#
#     feature_list[-2] = yuangao_type
#     feature_list[-1] = shuqing_type
#
#     return feature_list


def single_case_match(inputs, keyword_list, keyword_factor):  # todo
    """
    匹配单独的一个输入(多个句子)，得到特征
    :param input:
    :param problem:
    :return:
    """
    feature_list = np.zeros(len(keyword_list), dtype=int)  # 初始化特征序列
    sentence_factor = {}
    if inputs is None or len(inputs.strip()) == 0:
        return feature_list, sentence_factor

    inputs = remove_law_from_input(inputs)  # 去掉法条
    sentence_list = [s for s in re.split('[。；，：,;:]', inputs) if len(s) > 0]

    rest_keyword_list = {}
    for k, key_word in enumerate(keyword_list):
        if len(re.findall(key_word.replace('.*', '[^。；，：,;:]*'), inputs)) > 0:
            rest_keyword_list[key_word] = k

    for key_word, k in rest_keyword_list.items():  # 针对每一个关键词，设法去匹配
        for factor in keyword_factor[key_word]:
            for index, sentence in enumerate(sentence_list):  # 针对每个句子去匹配
                flag = match_sentence_by_keyword(sentence, factor, key_word)
                if flag == 1 or flag == -1:
                    feature_list[k] = flag
                    if key_word not in sentence_factor:
                        sentence_factor[key_word] = [sentence, [factor]]
                    else:
                        sentence_factor[key_word][1].append(factor)
                    break

    return feature_list, sentence_factor


def match_sentence_by_keyword(sentence, factor, key_word):
    """
    匹配关键词(key_word)。返回正向匹配（1）、负向匹配（-1）或没有匹配（0）
    :param sentence: 一整句话。如"xxx1，xxx2，xxx3，xxxx4。"
    :param key_word: 子案由因子下面的一个key_word
    :return:
    """
    if sentence.startswith('如') or sentence.startswith('若') or len(re.findall('[如若]' + key_word, sentence)) > 0:
        return 0
    if len(re.findall('(约定|要求|认为|应当|应该|如果).*' + key_word, sentence)) > 0:
        return 0
    flag_positive = len(re.findall(key_word, sentence)) > 0
    if key_word in no_positive_pattern and len(re.findall(no_positive_pattern[key_word], sentence)) > 0:
        return 0
    # 2.匹配负向
    if not flag_positive:
        return 0
    if key_word in no_negative_keyword:
        return 1
    if factor in no_negative_factor and len(re.findall(no_negative_factor[factor], key_word)) > 0:
        return 1
    for pattern in no_negative_pattern:
        if len(re.findall(pattern[0], key_word)) > 0 and len(re.findall(pattern[1], sentence)) > 0:
            return 1
    if len(re.findall(negative_word + '.*' + key_word, sentence)) > 0 \
            or len(re.findall(key_word + '.*' + negative_word, sentence)) > 0:
        #if negative_match(key_word, sentence) == -1:
        return -1
    if len(re.findall(negative_word, key_word)) == 0 and '.*' in key_word:
        kl = key_word.split('.*')
        for i in range(len(kl) - 1):
            kw = '.*'.join([k + '.*' + negative_word if i == j else k for j, k in enumerate(kl)])
            if len(re.findall(kw, sentence)) > 0:# and negative_match(key_word, sentence) == -1:
                return -1
    return 1


def negative_match(key_word, sentence):
    words = list(segmentor.segment(sentence))  # 分词 元芳你怎么看
    postags = list(postagger.postag(words))  # 词性标注
    arcs = parser.parse(words, postags)  # 句法分析
    for i in range(len(arcs)):
        if words[i] in negative_word_list and words[i] not in key_word and arcs[i].relation != 'HED':
            if words[arcs[i].head - 1] in key_word:
                return -1
            if arcs[arcs[i].head - 1].relation == 'VOB' and words[arcs[arcs[i].head - 1].head - 1] in key_word:
                return -1
        if words[i] in negative_word_list and words[i] not in key_word and arcs[i].relation == 'HED':
            for k, arc in enumerate(arcs):
                if arc.relation in ['SBV', 'VOB'] and arc.head == i + 1 and words[k] in key_word:
                    return -1
        if words[i] in negative_word_list and words[i] not in key_word and arcs[arcs[i].head - 1].relation == 'HED':
            for k, arc in enumerate(arcs):
                if arc.relation in ['SBV', 'VOB'] and arc.head == arcs[i].head and words[k] in key_word:
                    return -1
        if words[i] in key_word and words[arcs[i].head - 1] not in key_word and words[
            arcs[i].head - 1] in negative_word_list:
            return -1
    return 1


def get_input_by_type(line, input_type):
    """
    通过类型，获得需要的字段的值
    :param line: 源数据
    :param input_type: 类型（'fact','reason','result'）
    :return: 需要的数据
    """
    sub_cause_name, yuan_gao, bei_gao, su_cheng, beigaobiancheng, chamin, panjue, benyuanrenwei = line.strip().split(
        fact_reason_splitter)
    if random.randint(0, 500) == 1:  # 选择性打印一些原始数据的日子
        print("---------------------------------")
        print("yuan_gao:", yuan_gao)
        print("bei_gao:", bei_gao)
        print("su_cheng as input:", su_cheng, ';input_type:', input_type)
        print("benyuan_renwei:", benyuanrenwei)
        print("panjue:", panjue)

    input = ''
    if input_type == 'fact':
        input = su_cheng  # +" "+chamin
    elif input_type == 'reason':
        input = benyuanrenwei
    else:  # 默认是从'诉讼请求'预测'判决结果'
        input = su_cheng
    return input


def remove_law_from_input(inputs):
    """
    移除一些法律法规
    规则1： “”根据.*规定“. e.g.”根据《中华人民共和国劳动合同法》第三十条：“用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。”的规定，万雪请求金海煤矿支付劳动报酬的主张符合法律规定，本院应予支持。
    :param input:
    :return:
    """
    for pattern in re.findall('“.*?”', inputs):
        start_index = inputs.index(pattern)
        end_index = inputs.index(pattern) + len(pattern)
        inputs = inputs[:start_index] + inputs[end_index:]
    inputs = re.sub('(根据|依据)(.*?)的规定', '', inputs)
    # pattern_1 = '根据(.*?)的规定'
    # # for i,input in enumerate(input_list):
    # # print("input:",input,"match or not:",len(re.findall(pattern_1, input))>0)
    # if len(re.findall(pattern_1, inputs)) > 0:
    #     pattern_1 = re.compile(pattern_1)
    #     match_result_1 = re.match(pattern_1, inputs)
    #     if match_result_1 is not None:
    #         entity = match_result_1.group(1)
    #         # print("entity:",entity)
    #         inputs = inputs.replace(entity, "xxx")
    return inputs


# 打印运营打标的真实案例所需要的文件
def pk_print_what_my_need(id, sequence, factor_keywords, problem, suqiu):
    factor_keywords_list = list(factor_keywords.index)
    factor_keywords_str = '|'.join(factor_keywords_list)

    problem_suqiu_keyword_factor_dict = get_problem_suqiu_keyword_factor_dict()
    keyword_list = problem_suqiu_keyword_factor_dict[problem][suqiu].index.tolist()
    keyword_factor = problem_suqiu_keyword_factor_dict[problem][suqiu]
    keyword_list_expanded = keyword_list_expand(keyword_list)
    keyword_factor_expanded = {keyword_list_expand(k): v for k, v in keyword_factor.items()}
    feature_list, sentence_factor = single_case_match(sequence, keyword_list_expanded, keyword_factor_expanded, 'fact')

    matched_result = {}
    for v in sentence_factor.values():
        if v[0] not in matched_result:
            matched_result[v[0]] = v[1]
        else:
            matched_result[v[0]] += v[1]
    matched_result = {k: set(v) for k, v in matched_result.items()}
    # print(matched_result)
    matched_result_list = list(matched_result.keys())

    sss = ''
    for i in range(len(matched_result_list)):
        s1 = matched_result_list[i]
        s2 = list(matched_result[s1])
        s3 = '|'.join(s2)
        s4 = s1 + ':' + s3
        sss += s4
        sss += 'aaa'

    with open('./data/yy_marking.csv', 'a') as ff:
        sequence = sequence.replace(',', '，')
        need_data = str(id) + ',' + problem + ',' + suqiu + ',' + sequence + ',' + factor_keywords_str + ',' + sss
        need_data = need_data.replace('\n', '')
        need_data = need_data.replace(' ', '')
        need_data = need_data.replace('\t', '')
        need_data = need_data.replace('\r', '')
        need_data = need_data.replace('aaa', '\n')
        ff.write(need_data)
        ff.write('\n')


def pk_print_factor_yy_need_marking(id, sequence, factor_keywords, problem, suqiu):
    factor_keywords_list = list(factor_keywords.index)
    factor_keywords_str = ':,,'.join(factor_keywords_list)
    with open('./data/yy_factor_need.csv', 'a') as ff:
        sequence = sequence.replace(',', '，')
        sequence = sequence.replace('\n', '')
        sequence = sequence.replace(' ', '')
        sequence = sequence.replace('\t', '')
        sequence = sequence.replace('\r', '')
        need_data = str(id) + ',' + problem + ',' + suqiu + ',' + sequence + ',' + factor_keywords_str + ','
        ff.write(need_data)
        ff.write('\n')
