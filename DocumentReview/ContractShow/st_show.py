#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 13:27
# @Author  : Adolf
# @Site    : 
# @File    : st_show.py
# @Software: PyCharm
import streamlit as st
from DocumentReview.ParseFile.parse_word import read_docx_file

contract_type = st.sidebar.selectbox("请选择合同类型", ["借条", "借款合同", "劳动合同"], key="合同类型")
mode_type = st.sidebar.selectbox("请选择上传数据格式", ["text", "docx"], key="text")

if contract_type == '借条':
    from DocumentReview.ContractReview.loan_review import LoanUIEAcknowledgement

    acknowledgement = LoanUIEAcknowledgement(config_path="DocumentReview/Config/LoanConfig/jietiao_20220531.csv",
                                             log_level="info",
                                             model_path="model/uie_model/model_best/")
elif contract_type == '借款合同':
    from DocumentReview.ContractReview.loan_contract_review import LoanContractUIEAcknowledgement

    acknowledgement = LoanContractUIEAcknowledgement(
        config_path="DocumentReview/Config/LoanConfig/jiekuan_20220605.csv",
        log_level="info",
        model_path="model/uie_model/jkht/model_best")
elif contract_type == '劳动合同':
    from DocumentReview.ContractReview.labor_review import LaborUIEAcknowledgement

    acknowledgement = LaborUIEAcknowledgement(config_path="DocumentReview/Config/LaborConfig/labor_20220615.csv",
                                              log_level="info",
                                              model_path="model/uie_model/labor/model_best")
else:
    raise Exception("暂时不支持该合同类型")


@st.cache
def get_data(_file):
    _text = read_docx_file(_file)
    # print(_text)
    return "\n".join(_text)


def show_res(_text):
    acknowledgement.review_main(content=_text, mode="text")
    st.write("合同审核结果:")
    st.write(acknowledgement.review_result)


if mode_type == "text":
    if contract_type == '借条':
        text = st.text_area(label="请输入文本内容", height=600,
                            value="借 条\n为购买房产，今收到好友张三（身份证号）以转账方式出借的人民币壹万元整（￥10000.00元），\
             借期拾个月，月利率1%，于2023年05月23日到期时还本付息。逾期未还，则按当期一年期贷款市场报价利率（LPR）的4倍计付逾期利息。\n如任何一方（借款人、债务人）\
             违约，守约方（出借人、债权人）为维护权益向违约方追偿的一切费用（包括但不限于律师费、诉讼费、保全费、交通费、差旅费、鉴定费等等）均由违约方承担。\n身份证载明\
             的双方（各方）通讯地址可作为送达催款函、对账单、法院送达诉讼文书的地址，因载明的地址有误或未及时告知变更后的地址，\
             导致相关文书及诉讼文书未能实际被接收的、邮寄送达的，相关文书及诉讼文书退回之日即视为送达之日。\n借款人的微信号为：ffdsaf\n借款人：李四\n身份\
             证号：123132423142314231\n联系电话：13242314123\n借款人：李四媳妇\n身份证号：12343124\n联系电话：1342324123\n家庭住\
             址：（具体到门牌号）\n2022年05月12日", key="text")
    elif contract_type == '借款合同':
        text = st.text_area(label="请输入文本内容", height=600,
                            value="自然人之间的借款合同\n甲方（出借人）：陈铭生\n身份证号：330121198509120989 \
        \n联系地址：江苏省南京市玄武区临近街道21号\n乙方（借款人）：方茴\n身份证号：330121199209122398 \
        \n联系地址：浙江省杭州市西湖区文三路123号\n丙方（保证人）：陈寻\n身份证号：330121199309111232\n联系地址：浙江省杭 \
        州市西湖区文三路122号\n 乙方因买车 ，向甲方借款，丙方愿意为乙方借款向甲方提供连带保证担保，现甲乙丙各方在平等、自愿、等价\
        有偿的基础上，经友好协商，达成如下一致意见，供双方共同信守。\n一、借款条款\n第一条 借款用途。乙方因买车急需一笔资金周转，甲\
        方同意出借，但乙方如何使用借款，则与甲方无关。\n第二条 借款金额。乙方向甲方借款金额（大写）拾万元整（小写：￥100000元）乙\
        方指定的收款账户为：\n开户银行：招商银行杭州分行\n账户名称： 方茴\n账 号： 12876567651234565782\n第三条 借款期限。借款期\
        限1年，自2022年1月1日起（以甲方实际出借款项之日起算，乙方应另行出具收条）至2023年1月1日止，逾期未还款的，按第九条处理。\n第\
        四条 还款方式。应按照本协议规定时间主动偿还对甲方的欠款及利息。乙方到期还清所有本协议规定的款项后，甲方收到还款后将借\
        据交给乙方。甲方指定的还款账户为：\n开户银行：工商银行杭州分行\n账户名称：陈铭生\n账 号：1878765678567845671             。\
        \n第五条 借款利息。自支用借款之日起，按实际支用金额计算利息，约定的借款期内月利为1%，利息按月结算。到期还本，结息日为\
        每月1日，借款人需要在每一结息日支付利息，如果借款本金的最后一次偿还日不再结息日，则未付利息应利随本清。借款方如果\
        不按期还款付息，则每逾期一日按欠款金额的每日万分之八加收违约金。\n二、担保条款\n第六条 借款方自愿用自用奔驰车做抵押，到期不\
        能归还贷款方的贷款，贷款方有权处理抵押品。借款方到期如数归还贷款的，抵押权消灭。\n第七条 丙方自愿为乙方的借款提供连带责任保证\
        担保，保证期限为自乙方借款期限届满之日起二年。保证担保范围包括借款本金、逾期还款的违约金或赔偿金、甲方实现债权的费用（包括\
        但不限于诉讼费、律师费、差旅费等）。\n三、权利义务\n第八条 贷款方有权监督贷款使用情况，了解借款方的偿债能力等情况，借款方应该如\
        实\n提供有关的资料。借款方如不按合同规定使用贷款，贷款方有权收回部分贷款，并对违约部分参照银行规定加收罚息。（贷款方提前还款的，应\
        按规定减收利息。）\n四、逾期还款的处理\n第九条 乙方如逾期还款，除应承担甲方实现债权之费用（包括但不限于甲方支出之律师\
        费、诉讼费、差旅费等）外，还应按如下方式赔偿甲方之损失：逾期还款期限在30日以内的部分，按逾期还款金额每日千分之贰（2‰）的比例赔偿\
        甲方损失；超过30日以上部分，按照逾期还款金额每日千分之贰点伍（2.5‰）的比例赔偿甲方损失。\n第十条 前款约定的损失赔偿比例，系各方\
        综合各种因素确定。在主张该违约金时，甲方无须对其损失另行举证，同时双方均放弃《中华人民共和国合同法》第一百一十四条规定的违约金或\
        损失赔偿金调整请求权。\n五、合同争议的解决方式\n第十一条 本合同履行过程中发生的争议，由当事人对双方友好协商解决，也可由第三人调\
        解，协商或调解不成的，可由任意一方依法向出借方所在地人民法院起诉。\n第十二条 本合同自双方签章之日起生效。本合同一式3份，借款人、出\
        借人保证人各持1份。每份均具有同等法律效力。\n第十三条 本合同项下的一切形式的通知、催告均采用书面形式向本合同各方预留的\
        地址发送，如有地址变更，应及时通知对方，书面通知以发送之日起三日届满视为送达。\n甲方：陈铭生\n乙方：方茴\n连带保证人：陈寻 \
        \n 签订约日期：2022.1.1\n2022年    1  月   1   日", key="text")
    elif contract_type == '劳动合同':
        text = st.text_area(label="请输入文本内容", height=600,
                            value="劳动合同\n甲方瑞斯白有限责任公司\n地址浙江省舟山市定海区临城街道99号法定代表人(主要负责人)\n乙方吴涛性别男身份证\
                     号码330121199809091232现通信地址浙江省舟山市定海区临城街道232号联系电话13343432345为建立劳动关系，明确权利义务，依据劳动\
                     法、劳动合同法等有关法律规定，在平等自愿、协商一致的基础上，订立本合同。\n第一条本合同期限自2022年1月1日起至2024年1月1日止。其中\
                     试用期为2022年1月1日起至2022年2月1日止。\n第二条甲方根据工作需要，安排乙方在保洁工作岗位，乙方的工作任务为保洁，工作地点为浙江省舟\
                     山市定海区临城街道99号。经双方协商同意，甲方可以调换乙方的工种或岗位及工作地点。\n乙方应认真履行岗位职责，遵守各项规章制度，服从\
                     管理，按时完成工作任务。\n第三条乙方按甲方规定完成工作任务的，甲方于每月5日支付工资，支付的工资为9000元/月，其中试用期的工资为\
                     8000元/月。或者实行计件制，计件单价为。\n第四条工作时间和休息休假、社会保险、劳动保护、劳动条件和职业危害防护等按照法律法规、规章等规定执\
                     行。\n第五条双方解除或终止劳动合同应按法定程序办理，甲方为乙方出具终止、解除劳动合同的通知书或相关证明。符合法律、法规规定的，支付乙方经济\
                     补偿。\n第六条其他未尽事项按照国家及地方现行有关规定执行。\n第七条双方其他约定\n第八条本合同双方各执一份，涂改或未经授权代签无效。\n甲方签\
                     字(盖章)乙方签字\n签订时间：2022年1月1日\n劳动者合同文本已领签字", key="text")
    else:
        raise Exception("暂时不支持该合同类型")

    # show_res()


elif mode_type == "docx":
    file = st.file_uploader('上传文件', type=['docx'], key=None)

    text = get_data(file)
    # loan_acknowledgement.review_main(content=file_path, mode="docx")


else:
    raise Exception("暂不支持该格式")

run = st.button("开始审核合同ing", key="run")

if run:
    show_res(text)
