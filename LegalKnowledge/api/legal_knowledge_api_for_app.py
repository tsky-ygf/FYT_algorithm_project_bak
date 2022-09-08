#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
@Author  : inamori1932
@Time    : 2022/8/4 13:17 
@Desc    : 普法常识模块的接口
"""
import traceback
from loguru import logger
import requests
from flask import Flask
from flask import request
from Utils.http_response import response_successful_result, response_failed_result
from LegalKnowledge.core import legal_knowledge_service as service


app = Flask(__name__)


@app.route('/get_columns', methods=["get"])
def get_columns():
    return response_successful_result(service.get_columns())


def _get_short_content(content):
    try:
        short_content = str(content).split("。")[0].split(" ")[-1].split("\u3000")[-1]
        return short_content[:30] + "..."
    except Exception:
        return "..."


def _get_simple_news(news):
    return [
        {
            "id": item["id"],
            "title": str(item["title"]).strip().strip("."),
            "content": _get_short_content(item["content"])
        }
        for item in news
    ]


@app.route('/get_news_by_column_id', methods=["get"])
def get_news_by_column_id():
    column_id = request.args.get("column_id")
    if column_id:
        news = _get_simple_news(service.get_news_by_column_id(column_id))
        return response_successful_result(news, {"total_amount": len(news)})
    return response_failed_result("No parameter: column_id")


@app.route('/get_news_by_keyword', methods=["get"])
def get_news_by_keyword():
    keyword = request.args.get("keyword")
    if keyword:
        news = _get_simple_news(service.get_news_by_keyword(keyword))
        return response_successful_result(news, {"total_amount": len(news)})
    return response_failed_result("No parameter: keyword")


@app.route('/get_news_by_news_id', methods=["get"])
def get_news_by_news_id():
    news_id = request.args.get("news_id")
    if news_id:
        if int(news_id) == 10:
            news = [{
            "id": 10,
            "title": "农村建房，这些法律风险早知道！",
            "release_time": "2022-08-18",
            "content": "<div class=\"text\"><p style=\"text-indent: 2em;\">有山有水有后院，养花种草品香茶，这是许多人向往的田园生活。近年来翻建房屋热潮引发诸多农村建房施工合同纠纷。如何保障自家“休闲小院”顺利建成？今天为你解读农村建房的那些事儿。</p><p style=\"text-indent: 2em;\">案例一：</p><p style=\"text-indent: 2em;\">2021年4月5日，吴某与姚某签订委托盖房协议书，约定姚某包工包料为吴某盖平房。二人在结算时，对部分项目的费用产生了争议。吴某认为，姚某曾口头承诺免费给自己安装走道门，但实际未安装，应当退还相应费用；此外，姚某还承诺，化粪池包含于下水工程之中，因此不应属于增项，自己无需额外付费。据此，吴某不同意给付姚某化粪池工程款，还要求姚某返还多付的1.3万元。</p><p style=\"text-indent: 2em;\">关于安装走道门的问题，根据法律规定，对于增项是否约定为赠送项目发包方负有举证义务，即本案中的房主吴某。安装走道门是否属于赠送项目在合同中没有明确约定，且吴某并未提供相关证据予以证明，故法院对于赠送项目的主张不予支持。</p><p style=\"text-indent: 2em;\">关于化粪池是否属于增项的问题，由于协议书并未就化粪池项目进行约定，虽然双方承包方式为包工包料，但并无法解读出姚某需要承担该施工内容，且吴某也未提交相关证据予以证明，故法院认定化粪池属于增项部分，因双方无法就化粪池的沙子水泥费用及人工费达成一致，法院将酌情予以确定。最终，法院判令吴某支付姚某工程款28000元，驳回姚某、吴某其他诉讼请求。</p><p style=\"text-indent: 2em;\">实践中，农村建房施工合同普遍存在以下容易遗漏的风险点：一是对于施工方口头承诺赠送的施工项目，当事人往往会遗漏在合同中进行约定。二是部分项目与某些合同通常载明的施工项目具有连贯性，也极易被房主忽略；三是对于部分赠送施工项目的赠送内容不明确，例如垒院墙、平整院子，施工方称赠送该类项目的本意通常是免去人工费用，但房主却认为是包工包料，导致双方对费用产生分歧。</p><p style=\"text-indent: 2em;\">因此，对于建房施工中所有增减项，双方都应当明确记载于书面合同中，避免口头约定的出现，防止因合同事项约定不明导致争议事项在维权过程中无据可依。</p><p style=\"text-indent: 2em;\">案例二：</p><p style=\"text-indent: 2em;\">房东王某与包工头李某签订了《农村建房施工合同》，约定李某为王某建设二层楼房一栋，合同分四次给款。合同签订后，双方按照合同约定开始施工，过程中包工头李某发现工程预算少于实际工程款金额，故向王某提出要求提前预支下一阶段工程款，王某不同意，李某表示不再继续干了。随后，王某找了其他包工队完成建房，并且向法院起诉要求李某退还多支付的工程款并承担违约责任。</p><p style=\"text-indent: 2em;\">合同签订后，双方当事人应当依约履行合同，未经双方同意任何一方无权变更合同内容。本案中，双方当事人就合同给款期限做了明确具体的约定，李某以合同继续履行会超出预算为由要求王某预支工程款没有事实和法律依据，王某有权拒绝，李某以此为由拒绝继续施工于法无据。</p><p style=\"text-indent: 2em;\">关于双方费用结算问题，本案合同中对于给款进度的约定与工程量并非完全对应，在合同未完全履行的情况下，应当以实际工程量作为双方结算依据。法院经审理认定，李某已履行的合同内容少于王某已支付的合同款项，李某应当退还王某多支付的工程款，且因为李某的行为属于典型违约行为，法院判决李某还应当承担相应的违约责任。</p><p style=\"text-indent: 2em;\">施工过程中，房主与施工方应恪守诚实信用原则，依约履行合同，减少纠纷发生。在过程中如遇到需要解除合同的情况，双方当事人应当对已履行部分做好证据留存，以便提供证据、还原事实。</p><p style=\"text-indent: 2em;\">案例三：</p><p style=\"text-indent: 2em;\">赵某和张某签订《农村建房施工合同》，约定赵某为张某包工包料建筑正房五间。合同签订后，因为张某邻居反对，多次造成工人停工。后来，双方就继续履行合同达成补充协议，约定此后张某保证不再出现因张某原因延误工期的情况，并预支下一阶段工程进度款六万元，剩余尾款在房屋全部完工后一个月内付清，赵某确保在2021年9月前完工。补充协议签订后，赵某在9月底前顺利完工，但张某却迟迟未支付工程尾款。故赵某将张某诉至法院，请求张某支付工程尾款三万元，并承担应停工造成的停工损失。</p><p style=\"text-indent: 2em;\">本案中，该合同的履行确实因为张某的原因导致多次停工，但赵某却无权基于该补充协议向张某要求停工损失。因为双方在合同履行过程中，对于之前停工的问题已经通过补充协议的方式做了变更约定，签订补充协议时赵某并未因停工问题向张某主张赔偿，视为双方对之前的合同行为达成合意。赵某因张某未按期支付尾款，而重新主张张某停工造成的违约损失于法无据，法院不予支持。</p><p style=\"text-indent: 2em;\">在施工过程中，施工方与房主应做好施工节点以及延期原因与延期时间的记录。房屋修建是房主和施工方共同努力的结果，希望双方积极配合、依约履行，促进房屋顺利建成。</p><p> </p><p> </p> </div>\n\t         \n\t        ",
            "raw_content": "<div class=\"text\"><p style=\"text-indent: 2em;\">有山有水有后院，养花种草品香茶，这是许多人向往的田园生活。近年来翻建房屋热潮引发诸多农村建房施工合同纠纷。如何保障自家“休闲小院”顺利建成？今天为你解读农村建房的那些事儿。</p><p style=\"text-indent: 2em;\">案例一：</p><p style=\"text-indent: 2em;\">2021年4月5日，吴某与姚某签订委托盖房协议书，约定姚某包工包料为吴某盖平房。二人在结算时，对部分项目的费用产生了争议。吴某认为，姚某曾口头承诺免费给自己安装走道门，但实际未安装，应当退还相应费用；此外，姚某还承诺，化粪池包含于下水工程之中，因此不应属于增项，自己无需额外付费。据此，吴某不同意给付姚某化粪池工程款，还要求姚某返还多付的1.3万元。</p><p style=\"text-indent: 2em;\">关于安装走道门的问题，根据法律规定，对于增项是否约定为赠送项目发包方负有举证义务，即本案中的房主吴某。安装走道门是否属于赠送项目在合同中没有明确约定，且吴某并未提供相关证据予以证明，故法院对于赠送项目的主张不予支持。</p><p style=\"text-indent: 2em;\">关于化粪池是否属于增项的问题，由于协议书并未就化粪池项目进行约定，虽然双方承包方式为包工包料，但并无法解读出姚某需要承担该施工内容，且吴某也未提交相关证据予以证明，故法院认定化粪池属于增项部分，因双方无法就化粪池的沙子水泥费用及人工费达成一致，法院将酌情予以确定。最终，法院判令吴某支付姚某工程款28000元，驳回姚某、吴某其他诉讼请求。</p><p style=\"text-indent: 2em;\">实践中，农村建房施工合同普遍存在以下容易遗漏的风险点：一是对于施工方口头承诺赠送的施工项目，当事人往往会遗漏在合同中进行约定。二是部分项目与某些合同通常载明的施工项目具有连贯性，也极易被房主忽略；三是对于部分赠送施工项目的赠送内容不明确，例如垒院墙、平整院子，施工方称赠送该类项目的本意通常是免去人工费用，但房主却认为是包工包料，导致双方对费用产生分歧。</p><p style=\"text-indent: 2em;\">因此，对于建房施工中所有增减项，双方都应当明确记载于书面合同中，避免口头约定的出现，防止因合同事项约定不明导致争议事项在维权过程中无据可依。</p><p style=\"text-indent: 2em;\">案例二：</p><p style=\"text-indent: 2em;\">房东王某与包工头李某签订了《农村建房施工合同》，约定李某为王某建设二层楼房一栋，合同分四次给款。合同签订后，双方按照合同约定开始施工，过程中包工头李某发现工程预算少于实际工程款金额，故向王某提出要求提前预支下一阶段工程款，王某不同意，李某表示不再继续干了。随后，王某找了其他包工队完成建房，并且向法院起诉要求李某退还多支付的工程款并承担违约责任。</p><p style=\"text-indent: 2em;\">合同签订后，双方当事人应当依约履行合同，未经双方同意任何一方无权变更合同内容。本案中，双方当事人就合同给款期限做了明确具体的约定，李某以合同继续履行会超出预算为由要求王某预支工程款没有事实和法律依据，王某有权拒绝，李某以此为由拒绝继续施工于法无据。</p><p style=\"text-indent: 2em;\">关于双方费用结算问题，本案合同中对于给款进度的约定与工程量并非完全对应，在合同未完全履行的情况下，应当以实际工程量作为双方结算依据。法院经审理认定，李某已履行的合同内容少于王某已支付的合同款项，李某应当退还王某多支付的工程款，且因为李某的行为属于典型违约行为，法院判决李某还应当承担相应的违约责任。</p><p style=\"text-indent: 2em;\">施工过程中，房主与施工方应恪守诚实信用原则，依约履行合同，减少纠纷发生。在过程中如遇到需要解除合同的情况，双方当事人应当对已履行部分做好证据留存，以便提供证据、还原事实。</p><p style=\"text-indent: 2em;\">案例三：</p><p style=\"text-indent: 2em;\">赵某和张某签订《农村建房施工合同》，约定赵某为张某包工包料建筑正房五间。合同签订后，因为张某邻居反对，多次造成工人停工。后来，双方就继续履行合同达成补充协议，约定此后张某保证不再出现因张某原因延误工期的情况，并预支下一阶段工程进度款六万元，剩余尾款在房屋全部完工后一个月内付清，赵某确保在2021年9月前完工。补充协议签订后，赵某在9月底前顺利完工，但张某却迟迟未支付工程尾款。故赵某将张某诉至法院，请求张某支付工程尾款三万元，并承担应停工造成的停工损失。</p><p style=\"text-indent: 2em;\">本案中，该合同的履行确实因为张某的原因导致多次停工，但赵某却无权基于该补充协议向张某要求停工损失。因为双方在合同履行过程中，对于之前停工的问题已经通过补充协议的方式做了变更约定，签订补充协议时赵某并未因停工问题向张某主张赔偿，视为双方对之前的合同行为达成合意。赵某因张某未按期支付尾款，而重新主张张某停工造成的违约损失于法无据，法院不予支持。</p><p style=\"text-indent: 2em;\">在施工过程中，施工方与房主应做好施工节点以及延期原因与延期时间的记录。房屋修建是房主和施工方共同努力的结果，希望双方积极配合、依约履行，促进房屋顺利建成。</p><p> </p><p> </p> </div>\n\t         \n\t        ",
            "source_url": "https://bjgy.chinacourt.gov.cn/article/detail/2022/08/id/6860904.shtml"}]
        else:
            news = service.get_news_by_news_id(news_id)
        return response_successful_result(news[0] if news else dict())
    return response_failed_result("No parameter: news_id")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8122, debug=False)
