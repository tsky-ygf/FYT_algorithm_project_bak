# -*- coding: utf-8 -*-
import json
import traceback
import logging
import logging.handlers
import requests
from flask import Flask
from flask import request

from LawsuitPrejudgment.main.reasoning_graph_predict import predict_fn
from LawsuitPrejudgment.Administrative.administrative_api_v1 import *

"""
推理图谱的接口
"""
app = Flask(__name__)
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
logger.setLevel(logging.INFO)
handler = logging.handlers.TimedRotatingFileHandler('./service.log', when='D', interval=1)
handler.setFormatter(formatter)
logger.addHandler(handler)


def _request_parse(_request):
    '''解析请求数据并以json形式返回'''
    if _request.method == 'POST':
        return _request.json
    elif _request.method == 'GET':
        return _request.args
    else:
        raise Exception("传入了不支持的方法。")


@app.route('/get_civil_problem_summary', methods=["get"])
def get_civil_problem_summary():
    try:
        with open("LawsuitPrejudgment/main/civil_problem_summary.json") as json_data:
            problem_summary = json.load(json_data)["value"]
        return json.dumps({"success": True, "error_msg": "", "value": problem_summary}, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"success": False, "error_msg": repr(e), "value": None}, ensure_ascii=False)


@app.route('/get_template_by_problem_id', methods=["get"])
def get_template_by_problem_id():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": {
            "template": "男女双方自愿/不自愿（不自愿的原因）登记结婚，婚后育有x子/女，现 x岁， 因xx原因离婚。婚姻/同居期间，有存款x元、房屋x处、车子x辆、债务x元。（双方是否对子女、财产、债务等达成协议或已有法院判决，协议或判决内容，双方对协议或判决的履行情况）。"
        }}, ensure_ascii=False)
    pass


@app.route('/get_claim_list_by_problem_id', methods=["get"])
def get_claim_list_by_problem_id():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": [{
            "id": 461,
            "claim": "离婚"
        }, {
            "id": 462,
            "claim": "财产分割"
        }, {
            "id": 463,
            "claim": "返还彩礼"
        }]
    }, ensure_ascii=False)
    pass


@app.route('/get_claim_by_claim_id', methods=["get"])
def get_claim_by_claim_id():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": {
            "claim": "离婚"
        }
    }, ensure_ascii=False)
    pass


@app.route('/reasoning_graph_result', methods=["post"])  # "service_type":'ft'
def reasoning_graph_result():
    try:
        in_json = request.get_data()
        if in_json is not None:
            in_dict = json.loads(in_json.decode("utf-8"))
            problem = in_dict['problem']
            claim_list = in_dict['claim_list']
            fact = in_dict.get('fact', '')
            question_answers = in_dict.get('question_answers', {})
            factor_sentence_list = in_dict.get('factor_sentence_list', [])

            logging.info("=============================================================================")
            logging.info("1.problem: %s" % (problem))
            logging.info("2.claim_list: %s" % (claim_list))
            logging.info("3.fact: %s" % (fact))
            logging.info("4.question_answers: %s" % (question_answers))

            result_dict = predict_fn(problem, claim_list, fact, question_answers, factor_sentence_list)

            logging.info("5.result.result_dict: %s" % (result_dict))

            question_asked = result_dict['question_asked']  # 问过的问题
            question_next = result_dict['question_next']  # 下一个要问的问题
            question_type = result_dict['question_type']
            factor_sentence_list = result_dict['factor_sentence_list']  # 匹配到短语的列表
            result = result_dict['result']
            if len(result) == 0:
                result = None
            else:
                parsed_result = []
                for suqiu, report in result.items():
                    parsed_result.append({
                        "claim": suqiu,
                        "support_or_not": report.get("support_or_not"),
                        "possibility_support": report.get("possibility_support"),
                        "reason_of_evaluation": report.get("reason_of_evaluation"),
                        "evidence_module": report.get("evidence_module"),
                        "legal_advice": report.get("legal_advice"),
                        "applicable_law": [{
                            "law_name": "《中华人民共和国民法典》",
                            "law_item": "第一千零八十九条",
                            "law_content": "离婚时,夫妻共同债务应当共同偿还。共同财产不足清偿或者财产归各自所有的，由双方协议清偿;协议不成的，由人民法院判决。"
                        },
                            {
                                "law_name": "《最高人民法院关于适用《中华人民共和国婚姻法》若干问题的解释(二)》",
                                "law_item": "第十条",
                                "law_content": "当事人请求返还按照习俗给付的彩礼的，如果查明属于以下情形，人民法院应当予以支持：（一）双方未办理结婚登记手续的；（二）双方办理结婚登记手续但确未共同生活的；（三）婚前给付并导致给付人生活困难的。适用前款第（二）、（三）项的规定，应当以双方离婚为条件。"
                            }
                        ],
                        "similar_case": [{
                            "doc_id": "2b2ed441-4a86-4f7e-a604-0251e597d85e",
                            "similar_rate": "88%",
                            "title": "原告王某某与被告郝某某等三人婚约财产纠纷一等婚约财产纠纷一审民事判决书",
                            "court": "公主岭市人民法院",
                            "judge_date": "2016-04-11",
                            "case_number": "（2016）吉0381民初315号",
                            "tag": "彩礼 证据 结婚 给付 协议 女方 当事人 登记 离婚",
                            "win_or_not": True
                        },
                            {
                                "doc_id": "ws_c4b1e568-b253-4ac3-afd7-437941f1b17a",
                                "similar_rate": "80%",
                                "title": "原告彭华刚诉被告王金梅、王本忠、田冬英婚约财产纠纷一案",
                                "court": "龙山县人民法院",
                                "judge_date": "2011-07-12",
                                "case_number": "（2011）龙民初字第204号",
                                "tag": "彩礼 酒席 结婚 费用 订婚 电视 女方 买家 猪肉",
                                "win_or_not": False
                            }
                        ]
                    })
                result = parsed_result
            logging.info("6.service.result: %s" % (result))
            return json.dumps({
                "success": True,
                "error_msg": "",
                "question_asked": question_asked,
                "question_next": question_next,
                "question_type": question_type,
                "factor_sentence_list": factor_sentence_list,
                "result": result
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error_msg": "request data is none."
            }, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


@app.route('/get_law_document', methods=["get"])
def get_law_document():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "value": {
            "doc_id": "2b2ed441-4a86-4f7e-a604-0251e597d85e",
            "doc_title": "原告王某某与被告郝某某等三人婚约财产纠纷一等婚约财产纠纷一审民事判决书",
            "case_number": "（2016）吉0381民初315号",
            "judge_date": "2016-04-11",
            "province": "吉林省",
            "court": "公主岭市人民法院",
            "doc_content": "<p align=\"center\">公主岭市人民法院</p><p align=\"center\">民事判决书</p><p align=\"right\">（2016）吉0381民初315号</p><p>原告王某甲，男，现住吉林省公主岭市。</br>被告郝某甲，女，现住吉林省公主岭市。</br>被告郝某乙，男，现住吉林省公主岭市。</br>被告李某某，女，现住吉林省公主岭市。</p><p>原告王某甲与被告郝某甲、郝某乙、李某某婚约财产纠纷一案，本院受理后，依法组成合议庭，公开开庭进行了审理。王某甲、郝某甲、郝某乙、李某某到庭参加了诉讼。本案现已审理终结。</p><p>王某甲诉称：原告与郝某甲经自由恋爱于2015年1月14日登记结婚，双方未生育子女。2015年6月10日原告给被告过彩礼10万元及金戒指一枚，彩礼数额是10万元，当时原告把10万元放在郝某甲包里了，下午其父亲郝某乙把这钱存到银行。2015年7月1日原告与郝某甲依民俗举行了结婚仪式。2015年7月13日双方到民政部门办理了离婚登记手续，办离婚手续是为了买房子，现在我们双方不能和好了，离婚是自愿的。原告家庭非常困难，父亲又系低保户，因女方索要彩礼，导致原告家庭生活极度困难。原告与郝某甲结婚费用全是原告花的，女方没拿钱。郝某甲说怀孕流产这事原告不知道，说原告抢走郝某甲4万元钱也不属实。2015年1月14日原告与郝某甲登记结婚后，我们基本不在一起。现原告诉至法院：要求被告返还彩礼款人民币100000元；诉讼费用由被告承担。</p><p>郝某甲辩称：原告与2015年1月14日给了我压婚钱1万元。2015年6月中旬原告又给了我现金9万元。戒指给我了，是个白金的，价值也就1000元零点。我认为彩礼只包括现金9万元，别的不应包括，这9万元钱给我了，没给我父母，我存在我父亲银行卡里，因我没有当地银行卡。我俩是在2015年1月14日原告过生日那天登记结婚的，之后我俩一直同居，在南湖大路等地居住，搬了三四次家，房费都是我付的。我俩回门时，原告以买房为名义，从我这儿抢回去了4万元，说用于还外债。我俩共同生活期间，我俩没工作，我还怀孕了两次，都流产做下去了。我们的花销是我从借朋友的2万元，过完彩礼后，我直接还给我朋友了。我做人流花了4000多元多，在我家看病花销花了3000多元，另外还有租房子等费用。有一次原告冲我要钱，我不给，他打我，我爸为了不让我挨打，给了他1万元，原告打了收条。2015年9月份我俩分手的，没孩子。我怀孕两次，都流产作下去了。</br>李某某辩称：郝某甲说的都属实。他俩结婚住宾馆，我还给他们拿了3000元，雇车我们花了1000元。彩礼我没收到，是郝某甲收的。</br>郝某乙辩称：郝某甲、李某某说的都属实。</p><p>经审理查明，王某甲与郝媛艳系自由恋爱，于2015年1月14日登记结婚。2015年7月1日双方依民俗举行了婚礼仪式。2015年7月13日王某甲与郝媛艳在公主岭市民政局婚姻登记处办理了离婚登记手续，双方无子女。王某甲给付女方现金10万元，原、被告对其中现金9万元系彩礼款无异议。</p><p>本院认为，依照<a target='_blank' href='https://www.itslaw.com/search/lawsAndRegulations/lawAndRegulation?searchMode=lawsAndRegulations&lawAndRegulationId=1992f76d-78af-4042-aae9-811ad29866ce&lawRegulationArticleId=1000500846'>《最高人民法院关于民事诉讼证据的若干规定》第二条</a>规定：“当事人对自己提出的诉讼请求所依据的事实或者反驳对方诉讼请求所依据的事实有责任提供证据加以证明。没有证据或者证据不足以证明当事人的事实主张的，由负有举证责任的当事人承担不利后果”。<a target='_blank' href='https://www.itslaw.com/search/lawsAndRegulations/lawAndRegulation?searchMode=lawsAndRegulations&lawAndRegulationId=ce0ec7ce-f9fb-4722-b44e-8004c108250e&lawRegulationArticleId=1000367990'>《中华人民共和国婚姻法》第三条第一款</a>规定：“禁止包办、买卖婚姻和其他干涉婚姻自由的行为。禁止借婚姻索取财物”。《最高人民法院关于﹤中华人民共和国婚姻法﹥解释（二）》第十条第一款第三项的规定：“当事人请求返还按照习俗给付的彩礼的，如果查明属于以下情形，人民法院应当予以支持：（三）婚前给付并导致给付人生活困难的”。王某甲与郝某甲于2015年1月14日登记结婚，双方未生育子女。2015年7月13日王某甲与郝某甲在公主岭市民政局婚姻登记处协议离婚，现双方对于离婚均无异议。王某甲与郝某甲在离婚协议中对财产处理记载为“无”，并承诺同意协议内容，协议内容真实，如有虚假，愿承担一切法律责任。王某甲与郝某甲离婚时对财产处理的范围及方式均有义务进行真实、全面、明确约定，而按该离婚协议书所记载内容双方并未对财产问题进行处理，且王某甲亦未明确表示放弃索要彩礼的权利，庭审中王某甲、郝某甲、郝某乙、李某某又均承认结婚时原告给付了彩礼款。由于王某甲所过彩礼款数额较大，确给原告生活造成了困难，应酌情返还给部分彩礼款。郝某甲辩称王某甲给付的压婚钱1万元不属于彩礼款，由于依照民俗习惯压婚钱属于彩礼范畴，故王某甲给付郝某甲的彩礼款应认定为10万元。王某甲出庭证人王某乙、王某、刘某某虽证明当时彩礼款由郝某甲及其父母接受，但庭审中郝某乙、李某某予以否认，由于王某乙、王某、刘某某均系王某甲近亲属，且王某甲又未提供其他证据予以佐证，故其要求郝某乙、李某某承担返还彩礼责任的诉讼请求，本院无法予以支持。庭审中郝某甲提供了2014年10月18日检验单照片证明其曾经怀孕，王某甲又自称2015年7月13日办离婚手续是为了买房子，故双方同居时间应超出婚姻存续期间。鉴于王某甲与郝某甲同居生活时间较长且已登记结婚，郝某甲在筹办婚礼过程中及与王某甲共同生活期间确有一定合理花销的事实，在确定郝某甲返还彩礼款数额时予以适当考虑。对于郝某甲称王某甲抢走4万元的事实，由于其提供的证人均称此事系听郝某甲所诉并未在现场，且郝某丙、郝某丁等为郝某甲近亲属，故本院无法予以认定。对于郝某甲称王某甲拿走1万元的事实，其提供收据一张予以证明，质证中王某甲亦予认可，故应在返还彩礼数额中扣除。</br>综上所述，依据<a target='_blank' href='https://www.itslaw.com/search/lawsAndRegulations/lawAndRegulation?searchMode=lawsAndRegulations&lawAndRegulationId=ce0ec7ce-f9fb-4722-b44e-8004c108250e&lawRegulationArticleId=1000367990'>《中华人民共和国婚姻法》第三条第一款</a>、《最高人民法院关于﹤中华人民共和国婚姻法﹥解释（二）》第十条第一款第三项、<a target='_blank' href='https://www.itslaw.com/search/lawsAndRegulations/lawAndRegulation?searchMode=lawsAndRegulations&lawAndRegulationId=1992f76d-78af-4042-aae9-811ad29866ce&lawRegulationArticleId=1000500846'>《最高人民法院关于民事诉讼证据的若干规定》第二条</a>的规定，判决如下：</p><p>一、被告郝某甲于本判决生效后立即返还原告王某甲彩礼款人民币40000元。</br>二、驳回原告王某甲其他诉讼请求。</br>如果未按本判决指定的期间履行给付金钱义务，应当依照<a target='_blank' href='https://www.itslaw.com/search/lawsAndRegulations/lawAndRegulation?searchMode=lawsAndRegulations&lawAndRegulationId=d67a7d01-f233-4bfc-9227-b8cef1d4196b&lawRegulationArticleId=1000385579'>《中华人民共和国民事诉讼法》第二百五十三条</a>之规定，加倍支付迟延履行期间的债务利息。</br>案件受理费2300元由原告王某甲负担1150元，被告郝某甲负担1150元。</br>如不服本判决，可在判决书送达之日起十五日内，向本院递交上诉状，并按对方当事人的人数提出副本，上诉于吉林省四平市中级人民法院。</p><p>审判长杨君</br>审判员王锐</br>人民陪审员孙国印</p><p>二〇一六年四月十一日</p><p>书记员高荣莉</p>",
            "doc_segments": ["公主岭市人民法院", "民事判决书", "（2016）吉0381民初315号",
                             "原告王某甲，男，现住吉林省公主岭市。被告郝某甲，女，现住吉林省公主岭市。被告郝某乙，男，现住吉林省公主岭市。被告李某某，女，现住吉林省公主岭市。",
                             "原告王某甲与被告郝某甲、郝某乙、李某某婚约财产纠纷一案，本院受理后，依法组成合议庭，公开开庭进行了审理。王某甲、郝某甲、郝某乙、李某某到庭参加了诉讼。本案现已审理终结。",
                             "王某甲诉称：原告与郝某甲经自由恋爱于2015年1月14日登记结婚，双方未生育子女。2015年6月10日原告给被告过彩礼10万元及金戒指一枚，彩礼数额是10万元，当时原告把10万元放在郝某甲包里了，下午其父亲郝某乙把这钱存到银行。2015年7月1日原告与郝某甲依民俗举行了结婚仪式。2015年7月13日双方到民政部门办理了离婚登记手续，办离婚手续是为了买房子，现在我们双方不能和好了，离婚是自愿的。原告家庭非常困难，父亲又系低保户，因女方索要彩礼，导致原告家庭生活极度困难。原告与郝某甲结婚费用全是原告花的，女方没拿钱。郝某甲说怀孕流产这事原告不知道，说原告抢走郝某甲4万元钱也不属实。2015年1月14日原告与郝某甲登记结婚后，我们基本不在一起。现原告诉至法院：要求被告返还彩礼款人民币100000元；诉讼费用由被告承担。",
                             "郝某甲辩称：原告与2015年1月14日给了我压婚钱1万元。2015年6月中旬原告又给了我现金9万元。戒指给我了，是个白金的，价值也就1000元零点。我认为彩礼只包括现金9万元，别的不应包括，这9万元钱给我了，没给我父母，我存在我父亲银行卡里，因我没有当地银行卡。我俩是在2015年1月14日原告过生日那天登记结婚的，之后我俩一直同居，在南湖大路等地居住，搬了三四次家，房费都是我付的。我俩回门时，原告以买房为名义，从我这儿抢回去了4万元，说用于还外债。我俩共同生活期间，我俩没工作，我还怀孕了两次，都流产做下去了。我们的花销是我从借朋友的2万元，过完彩礼后，我直接还给我朋友了。我做人流花了4000多元多，在我家看病花销花了3000多元，另外还有租房子等费用。有一次原告冲我要钱，我不给，他打我，我爸为了不让我挨打，给了他1万元，原告打了收条。2015年9月份我俩分手的，没孩子。我怀孕两次，都流产作下去了。李某某辩称：郝某甲说的都属实。他俩结婚住宾馆，我还给他们拿了3000元，雇车我们花了1000元。彩礼我没收到，是郝某甲收的。郝某乙辩称：郝某甲、李某某说的都属实。",
                             "经审理查明，王某甲与郝媛艳系自由恋爱，于2015年1月14日登记结婚。2015年7月1日双方依民俗举行了婚礼仪式。2015年7月13日王某甲与郝媛艳在公主岭市民政局婚姻登记处办理了离婚登记手续，双方无子女。王某甲给付女方现金10万元，原、被告对其中现金9万元系彩礼款无异议。",
                             "本院认为，依照《最高人民法院关于民事诉讼证据的若干规定》第二条规定：“当事人对自己提出的诉讼请求所依据的事实或者反驳对方诉讼请求所依据的事实有责任提供证据加以证明。没有证据或者证据不足以证明当事人的事实主张的，由负有举证责任的当事人承担不利后果”。《中华人民共和国婚姻法》第三条第一款规定：“禁止包办、买卖婚姻和其他干涉婚姻自由的行为。禁止借婚姻索取财物”。《最高人民法院关于﹤中华人民共和国婚姻法﹥解释（二）》第十条第一款第三项的规定：“当事人请求返还按照习俗给付的彩礼的，如果查明属于以下情形，人民法院应当予以支持：（三）婚前给付并导致给付人生活困难的”。王某甲与郝某甲于2015年1月14日登记结婚，双方未生育子女。2015年7月13日王某甲与郝某甲在公主岭市民政局婚姻登记处协议离婚，现双方对于离婚均无异议。王某甲与郝某甲在离婚协议中对财产处理记载为“无”，并承诺同意协议内容，协议内容真实，如有虚假，愿承担一切法律责任。王某甲与郝某甲离婚时对财产处理的范围及方式均有义务进行真实、全面、明确约定，而按该离婚协议书所记载内容双方并未对财产问题进行处理，且王某甲亦未明确表示放弃索要彩礼的权利，庭审中王某甲、郝某甲、郝某乙、李某某又均承认结婚时原告给付了彩礼款。由于王某甲所过彩礼款数额较大，确给原告生活造成了困难，应酌情返还给部分彩礼款。郝某甲辩称王某甲给付的压婚钱1万元不属于彩礼款，由于依照民俗习惯压婚钱属于彩礼范畴，故王某甲给付郝某甲的彩礼款应认定为10万元。王某甲出庭证人王某乙、王某、刘某某虽证明当时彩礼款由郝某甲及其父母接受，但庭审中郝某乙、李某某予以否认，由于王某乙、王某、刘某某均系王某甲近亲属，且王某甲又未提供其他证据予以佐证，故其要求郝某乙、李某某承担返还彩礼责任的诉讼请求，本院无法予以支持。庭审中郝某甲提供了2014年10月18日检验单照片证明其曾经怀孕，王某甲又自称2015年7月13日办离婚手续是为了买房子，故双方同居时间应超出婚姻存续期间。鉴于王某甲与郝某甲同居生活时间较长且已登记结婚，郝某甲在筹办婚礼过程中及与王某甲共同生活期间确有一定合理花销的事实，在确定郝某甲返还彩礼款数额时予以适当考虑。对于郝某甲称王某甲抢走4万元的事实，由于其提供的证人均称此事系听郝某甲所诉并未在现场，且郝某丙、郝某丁等为郝某甲近亲属，故本院无法予以认定。对于郝某甲称王某甲拿走1万元的事实，其提供收据一张予以证明，质证中王某甲亦予认可，故应在返还彩礼数额中扣除。综上所述，依据《中华人民共和国婚姻法》第三条第一款、《最高人民法院关于﹤中华人民共和国婚姻法﹥解释（二）》第十条第一款第三项、《最高人民法院关于民事诉讼证据的若干规定》第二条的规定，判决如下：",
                             "一、被告郝某甲于本判决生效后立即返还原告王某甲彩礼款人民币40000元。二、驳回原告王某甲其他诉讼请求。如果未按本判决指定的期间履行给付金钱义务，应当依照《中华人民共和国民事诉讼法》第二百五十三条之规定，加倍支付迟延履行期间的债务利息。案件受理费2300元由原告王某甲负担1150元，被告郝某甲负担1150元。如不服本判决，可在判决书送达之日起十五日内，向本院递交上诉状，并按对方当事人的人数提出副本，上诉于吉林省四平市中级人民法院。",
                             "审判长杨君审判员王锐人民陪审员孙国印", "二〇一六年四月十一日", "书记员高荣莉"]
        }
    }, ensure_ascii=False)
    pass


@app.route('/get_administrative_type', methods=["get"])
def get_administrative_type():
    # mock data
    return json.dumps({
        "success": True,
        "error_msg": "",
        "result": [{
            "type_id": "tax",
            "type_name": "税务处罚预判"
        }, {
            "type_id": "police",
            "type_name": "公安处罚预判"
        }, {
            "type_id": "transportation",
            "type_name": "道路运输处罚预判"
        }]
    }, ensure_ascii=False)
    pass


@app.route('/get_administrative_problem_and_situation_by_type_id', methods=["get", "post"])
def get_administrative_problem_and_situation_by_type_id():
    try:
        req_data = _request_parse(request)
        administrative_type = req_data.get("type_id")
        situation_dict = get_administrative_prejudgment_situation(administrative_type)
        # 编排返回参数的格式
        result = []
        for problem, value in situation_dict.items():
            situations = []
            for specific_problem, its_situations in value.items():
                situations.extend(its_situations)
            result.append({
                "problem": problem,
                "situations": situations
            })

        return json.dumps({
            "success": True,
            "error_msg": "",
            "result": result,
        }, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


@app.route('/get_administrative_result', methods=["get", "post"])
def get_administrative_result():
    try:
        req_data = _request_parse(request)
        administrative_type = req_data.get("type_id")
        situation = req_data.get("situation")
        res = get_administrative_prejudgment_result(administrative_type, situation)
        return json.dumps({
            "success": True,
            "error_msg": "",
            "result": res,
        }, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


@app.route('/get_criminal_result', methods=["post"])
def get_criminal_result():
    try:
        req_data = _request_parse(request)
        question = req_data.get("question")
        # 调用刑事预判的接口，获取结果
        url = "http://172.19.82.198:5060/get_criminal_report"
        data = {
            "question": question
        }
        resp_json = requests.post(url, json=data).json()

        # 编排接口返回内容的格式
        accusation = []
        for item in eval(resp_json.get("accusation")):
            for crime, prob in item.items():
                accusation.append({
                    "crime": crime,
                    "probability": prob
                })
        articles = []
        for item in eval(resp_json.get("articles")):
            articles.append({
                "law_name": item[0],
                "law_item": item[1],
                "crime": item[2],
                "law_content": item[3],
                "prob": item[4]
            })
        result = {
            "accusation": accusation,
            "articles": articles,
            "imprisonment": int(resp_json.get("imprisonment"))
        }

        return json.dumps({
            "success": True,
            "error_msg": "",
            "result": result,
        }, ensure_ascii=False)
    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({
            "success": False,
            "error_msg": "unknown error:" + repr(e)
        }, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5090, debug=True)  # , use_reloader=False)
