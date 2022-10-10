#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    : 
# @File    : server.py
# @Software: PyCharm
import _io
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from DocumentReview.server_use.contract_for_server import get_support_contract_types, init_model, get_user_standpoint, \
    get_text_from_file_link_path

app = FastAPI()

acknowledgement = init_model()


@app.get('/get_contract_type')
async def _get_support_contract_types():
    return {"result": get_support_contract_types()}


@app.get('/get_user_standpoint')
async def _get_user_standpoint():
    return {"result": get_user_standpoint()}


class ContractInput(BaseModel):
    contract_type_id: str = "laowu"
    usr: str = "party_a"
    contract_content: str = "劳务雇佣合同甲⽅（⽤⼈单位）：太极信息有限公司地址：浙江省杭州市滨江区物联网大街222号电话：13662677827⼄⽅（员⼯）姓" \
                            "名：陈斌性别：女年龄：34⽂化程度：高中电话：13444327837住址：浙江省杭州市萧山区鸿宁路23号⾝份证号：320102199709" \
                            "120098根据国家《劳动法》、《劳动合同法》和甲⽅的劳动⽤⼯管理等制度规定，遵循合法、公平、平等⾃愿、诚实信⽤的原则，经甲⼄" \
                            "双⽅协商⼀致，同意签订本劳动合同，共同遵守各严格履⾏约定的义务。第⼀条劳动合同期限本劳动合同期限从2022年3⽉5⽇起⾄2025" \
                            "年3⽉5⽇⽌，共3年。其中试⽤期为1⽉（⾃2022年3⽉5⽇起⾄2022年4⽉5⽇⽌）。第⼆条⼯作内容⼄⽅同意在甲⽅从事销售⼯作（岗" \
                            "位或⼯种）。第三条⽣产（⼯作）任务（岗位）甲⽅安排⼄⽅在⽣产部所辖范围内的相关岗位上⼯作（暂定），按照确定的具体岗位（任务）" \
                            "履⾏职责。⼄⽅同意甲⽅的安排，并按甲⽅的规定完成⽣产⼯作任务。甲⽅根据⽣产⼯作需要或⼄⽅的实际情况，可以调整⼄⽅的⼯作岗位，" \
                            "⼄⽅应当服从调整。如⼄⽅⽆合法正当理由拒不服从调整，则按⼄⽅⾃动解除劳动合同处理。第四条劳动保护和劳动条件甲⽅根据国家有关" \
                            "职⼯安全⽣产、劳动保护、职业卫⽣健康的规定，为⼄⽅提供必要的⽣产⼯作条件和劳动保护设施⽤品，保障⼄⽅的安全和健康。第五条" \
                            "⼯作时间和休息休假甲⽅依据国家规定确定⼄⽅的⼯作时间和休息、休假，但对公司实⾏不定时⼯作制和实⾏计件制⼯资员⼯执⾏综合计" \
                            "算⼯作时间的，遇公休⽇上班⼯作，属于正常上班，不发加班⼯资，公司视情在停⽔、停电、原料缺乏等停产时间或⽣产淡季合理安排员" \
                            "⼯补休，以保证计件制员⼯的休假权利。第六条劳动报酬甲⽅实⾏综合计算⼯时⼯作制（全额计件制或定额加超产），按照兼顾双⽅利益的" \
                            "原则制定科学、合理的劳动定额和计件单价。⼄⽅出勤达到法定⼯作天数，并完成定额任务，甲⽅根据⼄⽅相应岗位计件⼯资单价计算⼯资，" \
                            "按⽉以货币形式发给⼄⽅⼯资报酬。⼄⽅在正常上班完成任务后，甲⽅给付⼄⽅的⼯资不得低于当地政府规定的最低⼯资标准。第七条安全" \
                            "⽣产⼄⽅必须严格遵守安全⽣产操作规程，积极参加甲⽅组织的与本岗位相关的各种培训，依法签订《安全⽣产责任书》。⼄⽅在合同期间" \
                            "因违反安全操作规程，应接受甲⽅的处理，若给甲⽅造成经济损失的，应按相关规定给予赔偿。第⼋条社会保险。甲⽅按照国家法律和当地" \
                            "法规、政策规定，为⼄⽅办理社会保险。甲⼄双⽅按有关规定缴纳各⾃应承担的保险费。⼄⽅应缴纳的社会保险费由甲⽅按⽉在其⼯资中" \
                            "代扣代缴。终⽌或解除劳动合同时，社会保险关系即⾏终⽌，由⼄⽅⾃⾏接续或按照规定办理有关⼿续。第九条绩效考核。甲⽅根据公司规" \
                            "定对⼄⽅的⼯作任务完成情况进⾏考核，根据甲⽅效益和⼄⽅业绩，以及对⼄⽅的考核情况，决定⼄⽅职务和⼯资的升降。第⼗条劳动纪律" \
                            "甲⽅根据劳动法规，制定各项规章制度（员⼯⼿册）及甲⽅与⼄⽅签订的劳动安全⽣产岗位责任书（协议书）、保密协议，视为本合同的附" \
                            "件，⼄⽅必须⾃觉遵守甲⽅⽣产（⼯作）操作规程和规章制度，⾃觉维护甲⽅利益，服从甲⽅的领导和管理。如⼄⽅严重违反公司劳动纪律或" \
                            "规章制度，甲⽅将按严重违纪处理，解除其劳动合同。第⼗⼀条劳动合同的终⽌、解除、续订甲⽅或⼄⽅因故提前解除劳动合同，应提前3" \
                            "0⽇以书⾯形式通知对⽅（甲⽅按《劳动法》第25条和《劳动合同法》第39条规定解除⼄⽅合同的除外），本合同期满后⾃⾏终⽌。根据⽣产" \
                            "⼯作需要，经甲⼄双⽅协商⼀致后可以续签。第⼗⼆条违反劳动合同的责任合同期间，任何⼀⽅违反合同有关条款规定，给对⽅造成损失的，" \
                            "应视后果和责任⼤⼩，由违约⽅承担赔偿⾦。如发⽣劳动争议，双⽅可向本公司劳动争议调解委员会（调解⼩组）申请调解，如调解不成的，" \
                            "可⾃发⽣劳动争议之⽇起60⽇内向当地劳动仲裁机关申诉。第⼗三条本合同如有与国家法律法规相抵触的，则按国家相关的规定处理。第⼗四" \
                            "条甲、⼄双⽅需要协商约定的其他内容。 第⼗五条本合同⼀式⼆份经甲⼄双⽅签字（盖章）后⽣效，甲⼄双⽅各执⼀份。甲⽅盖章：太极信息" \
                            "有限公司⼄⽅签字：陈斌代表⼈签字：马西西2022年3⽉1⽇"


@app.post("/get_contract_review_result")
async def _get_contract_review_result(contract_input: ContractInput):
    acknowledgement.review_main(content=contract_input.contract_content, mode="text",
                                      contract_type=contract_input.contract_type_id, usr=contract_input.usr)
    # print("review_result review_result",time.localtime(), acknowledgement.review_result)
    # print("resresres", time.localtime() ,res)
    return {"result": acknowledgement.return_result}


class FileLinkInput(BaseModel):
    file_path: str = "https://nblh-fyt.oss-cn-hangzhou.aliyuncs.com/fyt/20220916/55712bdc-694a-438f-bcb0-1f5b66dd9bb5.docx"


@app.post("/get_text_from_file_link_path")
async def _get_text_from_file_link_path(file_link_input: FileLinkInput):
    return {"result": get_text_from_file_link_path(file_link_input.file_path)}


# class FileInput(BaseModel):
#     file: _io.BufferedReader = open('data/uploads/upload.docx','rb')


# @app.post("/get_text_from_file")
# async def _get_text_from_file(file_input: FileInput):
#     return {"return":get_text_from_file(file_input.file)}


if __name__ == "__main__":
    # 日志设置
    uvicorn.run('OnlineServer.ContractReview.server:app', host="0.0.0.0", port=8112, reload=False, workers=1)
