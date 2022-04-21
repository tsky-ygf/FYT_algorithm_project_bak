# -*- coding: utf-8 -*-
import jieba
import re
import datetime
import pandas as pd
from elasticsearch import Elasticsearch


es = Elasticsearch()
LAW_ES_IDX_NAME = 'law_content_search'
CASE_ES_IDX_NAME = 'case_content_search'
properties = ['宪法法律', '修正案', '法律解释', '司法解释及文件', '有关法律问题的决定', '行政法规及文件', '部委规章及文件']


##############################################################################################################################################
#
# 法条搜索
#
##############################################################################################################################################

def _keyword_mark(content, keywords, strict=False):
    keywords = keywords.replace('<font color="red">', '').replace('</font>', '')

    if len(keywords) < 4:
        return content.replace(keywords, '<font color="red">' + keywords + '</font>')

    if strict:
        words = jieba.lcut(keywords)
        for length in range(len(words),0,-1):
            for i in range(len(words)-length+1):
                word = ''.join(words[i:i+length])
                if len(word)>1:
                    content = content.replace(word, '<font color="red">' + word + '</font>')
    else:
        content = content.replace(keywords, '<font color="red">' + keywords + '</font>')
        for i in range(1, len(keywords)):
            word = keywords[:i]+'.{1}'+keywords[i:]
            content = re.sub(word, lambda x: '<font color="red">'+x.group(0)+'</font>', content)
    return content


def law_content_search(keywords):
    body = {
        "size": 1000,
        "min_score": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": keywords,
                        "fields": ["law_name^10", "chapter^5", "content_simple"],
                        "type": "phrase",
                        "slop": 1,
                    },
                },
            }
        },
        "_source": ['law_name', 'chapter', 'clause', 'content', 'publish_date', 'property', 'serials', 'index'],
    }
    res = es.search(index=LAW_ES_IDX_NAME, body=body)  # res = es.search(index="test-index", body={"query": {"match_all": {}}})
    content = {property: {} for property in properties}
    score = {property: {} for property in properties}
    for hit in res['hits']['hits']:
        key = hit['_source']['law_name']+'###'+hit['_source']['publish_date']
        value = [[hit['_source']['chapter'], hit['_source']['clause'], hit['_source']['serials'], hit['_source']['content'], hit['_source']['index']]]
        content[hit['_source']['property']][key] = content[hit['_source']['property']].get(key, []) + value
        score[hit['_source']['property']][key] = score[hit['_source']['property']].get(key, 0) + hit['_score']

    result = []
    count = 0
    for property in properties:
        _content = content[property]
        _score = score[property]
        if len(_content)>0:
            temp = {'property': property, 'values': []}
            keys = sorted(_score.items(), key=lambda x: x[1], reverse=True)

            for k in keys:
                vs = sorted(_content[k[0]], key=lambda x: x[4])
                temp['values'].append({'law_name': _keyword_mark(k[0].split('###')[0], keywords),
                                       'publish_date': k[0].split('###')[1],
                                       'content': [{'chapter': _keyword_mark(v[0].replace('\u3000', '、'),keywords),
                                                    'clause': v[1],
                                                    'serials': v[2],
                                                    'content': _keyword_mark(v[3].replace('\n', '<br>'), keywords)}
                                                   for i, v in enumerate(vs)]})
                count += len(vs)
            result.append(temp)
    return result, count


def law_content_search_test(keywords):
    return [{"property": "宪法法律",
             "values":[
                 {"law_name": "中华人民共和国婚姻法",
                  "publish_date": "1980-09-10",
                  "content": [
                      {"clause": "第一条",
                       "chapter": "第一章、总则",
                       "serials": "（2014）甬镇民初字第44号|（2014）开民一初字第21号",
                       "content": "本法是婚姻家庭关系的基本准则。"},
                      {"clause": "第二条",
                       "chapter": "第一章、总则",
                       "serials": '',
                       "content": "实行婚姻自由、一夫一妻、男女平等的婚姻制度。保护妇女、儿童和老人的合法权益。实行计划生育。"},
                      {"clause": "第三条",
                       "chapter": "第一章、总则",
                       "serials": "（2014）甬镇民初字第44号",
                       "content": "禁止包办、买卖婚姻和其他干涉婚姻自由的行为。<br>禁止借婚姻索取财物。禁止重婚。禁止家庭成员间的虐待和遗弃。"}
                  ]}
             ]}], 3


def law_content_search_by_name(keywords, law_name, property, publish_date):
    body = {
        "size": 1000,
        "query": {
            "bool": {
                "must": {
                    "match_phrase": {"law_name": law_name},
                },
                "must": {
                    "match_phrase": {"property": property},
                },
                "must":{
                    "term": {"publish_date": publish_date},
                }
            }
        },
        "_source": ['law_name', 'chapter', 'clause', 'property', 'content', 'publish_date'],
        "sort": [
            {"index": {"order": "asc"}}
        ]
    }
    res = es.search(index=LAW_ES_IDX_NAME, body=body)  # res = es.search(index="test-index", body={"query": {"match_all": {}}})

    result = {'law_name': law_name,
              'property': property,
              'publish_date': publish_date,
              'content': []}
    for hit in res['hits']['hits']:
        if hit['_source']['law_name']!=law_name:
            continue
        if hit['_source']['property']!=property:
            continue
        if hit['_source']['publish_date']!=publish_date:
            continue
        result['content'].append({'chapter': hit['_source']['chapter'],
                                  'clause': hit['_source']['clause'],
                                  'content': _keyword_mark(hit['_source']['content'].replace('\n', '<br>'), keywords)})
    return result, len(result['content'])


def law_content_search_by_name_test(keywords, law_name, property, publish_date):
    return {"law_name": "中华人民共和国婚姻法",
             "property": "宪法法律",
             "publish_date": "1980-09-10",
             "content": [
                 {"clause": "第一条",
                  "chapter": "第一章、总则",
                  "content": "本法是婚姻家庭关系的基本准则。"},
                 {"clause": "第二条",
                  "chapter": "第一章、总则",
                  "content": "实行婚姻自由、一夫一妻、男女平等的婚姻制度。保护妇女、儿童和老人的合法权益。实行计划生育。"},
                 {"clause": "第三条",
                  "chapter": "第一章、总则",
                  "content": "禁止包办、买卖婚姻和其他干涉婚姻自由的行为。<br>禁止借婚姻索取财物。禁止重婚。禁止家庭成员间的虐待和遗弃。"}]
            }, 3


##############################################################################################################################################
#
# 案例搜索
#
##############################################################################################################################################

size_per_page = 20


def case_content_search(keywords, current_page):
    current_page = int(current_page)
    result = []
    for ks in re.split('[,，；;]', keywords):
        # 案号搜索
        if len(re.findall('（[\d]*）.*\d号', ks))>0:
            result += case_content_search_by_serial(ks)
        # 法条搜索
        if len(re.findall('第.*条|法$', ks))>0:
            result += case_content_search_by_law_name(ks, current_page)
        # 省市搜索
        elif len(re.findall('[省市]', ks))>0:
            result += case_content_search_by_province(ks, current_page)
        # 纠纷类型搜索
        elif len(re.findall('纠纷', ks))>0:
            result += case_content_search_by_anyou(ks, current_page)
        # 关键词搜索
        else:
            result += case_content_search_by_keyword(ks, current_page)
    return result


def case_content_search_by_serial(serial):
    result = []
    body = {
        "size": 1,
        "query": {
            "term": {"serial": serial}
        },
        "_source": ['serial', 'title', 'province', 'court', 'anyou', 'date', 'content'],
    }
    res = es.search(index=CASE_ES_IDX_NAME, body=body)
    if len(res['hits']['hits'])>0:
        hit = res['hits']['hits'][0]
        result.append({'serial': hit['_source']['serial'],
                       'title': hit['_source']['title'],
                       'anyou': hit['_source']['anyou'],
                       'province': hit['_source']['province'],
                       'court': hit['_source']['court'],
                       'date': hit['_source']['date'],
                       'content': hit['_source']['content']})
    return result


def case_content_search_by_law_name(law_name, current_page):
    body = {
        "from": current_page * size_per_page,
        "size": size_per_page,
        "min_score": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": law_name,
                        "fields": ["province", "court"],
                        "type": "phrase",
                        "slop": 1,
                    },
                },
            }
        },
        "_source": ['serial', 'title', 'province', 'court', 'anyou', 'date', 'content'],
        "sort": [
            {"_score": {"order": "desc"}},
            {"date": {"order": "desc"}},
        ]
    }
    result = []
    res = es.search(index=CASE_ES_IDX_NAME, body=body)
    for hit in res['hits']['hits']:
        result.append({'serial': hit['_source']['serial'],
                       'title': hit['_source']['title'],
                       'anyou': hit['_source']['anyou'],
                       'province': hit['_source']['province'],
                       'court': hit['_source']['court'],
                       'date': hit['_source']['date'],
                       'content': _keyword_mark(hit['_source']['content'], law_name, False)})
    return result


def case_content_search_by_province(province, current_page):
    body = {
        "from": current_page * size_per_page,
        "size": size_per_page,
        "min_score": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": province,
                        "fields": ['fatiao'],
                        "type": "phrase",
                        "slop": 1,
                    },
                },
            }
        },
        "_source": ['serial', 'title', 'province', 'court', 'anyou', 'date', 'content'],
        "sort": [
            {"_score": {"order": "desc"}},
            {"date": {"order": "desc"}},
        ]
    }
    result = []
    res = es.search(index=CASE_ES_IDX_NAME, body=body)
    for hit in res['hits']['hits']:
        result.append({'serial': hit['_source']['serial'],
                       'title': hit['_source']['title'],
                       'anyou': hit['_source']['anyou'],
                       'province': _keyword_mark(hit['_source']['province'], province, False),
                       'court': _keyword_mark(hit['_source']['court'], province, False),
                       'date': hit['_source']['date'],
                       'content': _keyword_mark(hit['_source']['content'], province, False)})
    return result


def case_content_search_by_anyou(anyou, current_page):
    body = {
        "from": current_page * size_per_page,
        "size": size_per_page,
        "min_score": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": anyou,
                        "fields": ['anyou'],
                        "type": "phrase",
                        "slop": 1,
                    },
                },
            }
        },
        "_source": ['serial', 'title', 'province', 'court', 'anyou', 'date', 'content'],
        "sort": [
            {"_score": {"order": "desc"}},
            {"date": {"order": "desc"}},
        ]
    }
    result = []
    res = es.search(index=CASE_ES_IDX_NAME, body=body)
    for hit in res['hits']['hits']:
        result.append({'serial': hit['_source']['serial'],
                       'title': _keyword_mark(hit['_source']['title'], anyou, False),
                       'anyou': _keyword_mark(hit['_source']['anyou'], anyou, False),
                       'province': hit['_source']['province'],
                       'court': hit['_source']['court'],
                       'date': hit['_source']['date'],
                       'content': _keyword_mark(hit['_source']['content'], anyou, False)})
    return result


def case_content_search_by_keyword(keyword, current_page):
    body = {
        "from": current_page * size_per_page,
        "size": size_per_page,
        "min_score": 20,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": keyword,
                        "fields": ["title^3", "anyou^3", "sucheng^2", "content"],
                        'operator': 'and'
                    },
                },
            }
        },
        "_source": ['serial', 'title', 'province', 'court', 'anyou', 'date', 'content'],
        "sort": [
            {"_score": {"order": "desc"}},
            {"date": {"order": "desc"}},
        ]
    }
    result = []
    res = es.search(index=CASE_ES_IDX_NAME, body=body)
    for hit in res['hits']['hits']:
        result.append({'serial': hit['_source']['serial'],
                       'title': _keyword_mark(hit['_source']['title'], keyword, False),
                       'anyou': _keyword_mark(hit['_source']['anyou'], keyword, False),
                       'province': _keyword_mark(hit['_source']['province'], keyword, False),
                       'court': _keyword_mark(hit['_source']['court'], keyword, False),
                       'date': hit['_source']['date'],
                       'content': _keyword_mark(hit['_source']['content'], keyword, False)})
    return result


def case_content_search_test(keywords):
    return [{'title': '郑某与凌某离婚纠纷一审民事判决书',
             'anyou': '离婚纠纷',
             'province': '浙江省',
             'court': '宁波市镇海区人民法院',
             'date': '2014-01-20',
             'content': '<p align="center">宁波市镇海区人民法院</p><p align="center">民事判决书</p><p align="right">（2014）甬镇民初字第44号</p><p>原告：郑某。</br>委托代理人：杨惠康。</br>被告：凌某。</br>委托代理人：徐萍。</p><p>原告郑某与被告凌某离婚纠纷一案，本院于2013年12月26日立案受理后，依法由代理审判员贾毅飞适用简易程序独任审判，于2014年1月16日公开开庭进行了审理。原告郑某及其委托代理人杨惠康、被告凌某及其委托代理人徐萍到庭参加诉讼。本案现已审理终结。</p><p>原告郑某起诉称：原告于2006年经长辈介绍与被告认识，见过数面后，原告就向长辈提及两人不合适，包括生活习惯、消费观念以及业余爱好等。但因双方父亲曾经是同学，所以极力撮合此事，原告受长辈劝解的影响，最终接受，于2007年1月15日到镇海区招宝山街道领取结婚证。由于婚前双方缺乏了解，感情基础薄弱，沟通困难，婚后两人因各种原因时常有大小争吵。主要原因有：1.夫妻生活不正常。婚后夫妻生活不正常，一直未生育，碍于亲情压力，2012年1月，在杭州妇儿医院接受试管婴儿手术，同年10月16日，诞下双胞胎，儿子郑某某甲，女儿郑某某乙。但是孩子的到来反而突显了双方生活习惯、教育理念的矛盾，由此引发的争吵越来越频繁。2.经济问题。结婚以来双方经济都是独立的，家里的生活支出基本都是AA制。遇到买车、买房等大件支出，总有无休止的争吵、矛盾。3.生活习惯、业余爱好差异大的问题。因被告在安徽巢湖出生，一直到毕业后才来到宁波工作。所以双方在生活习惯上差异较大，而且对于孩子的教育观念也很不一样，由此经常引发大吵。而双方因性格、观念差异，沟通方式匮乏，很难达成一致。以上种种原因，导致争吵不断，怨恨日渐加重。由于无法忍受这种没有感情、没有温暖的夫妻生活，原、被告曾于2013年5月到民政局协议离婚，被告要求抚养两个子女，而原告认为被告在镇海区骆驼街道上班，其父母住址在北仑区，住宅面积约60平方，父母医疗等关系尚在安徽巢湖，需要定期返回核销医疗费用，存在很多不便因素，所以原告认为被告抚养两个孩子有困难，不利于孩子更好地成长。而原告在镇海上班，父母住在镇海繁景小区，住宅面积约90平方，附近生活、教育等配套设施成熟，利于孩子成长，所以原告希望双方各自抚养一个。因双方在孩子抚养权方面无法达成一致，所以无法成功协议离婚。原告也于2013年5月开始一直住在父母家，被告租住在骆驼，双方已是分居生活。期间双方家长、亲戚也曾试图调解，在详细了解情况后也不再提出反对意见。因原、被告双方没有感情基础，婚后夫妻感情一直未能建立，而且呈进一步恶化趋势，现夫妻感情完全、彻底破裂，夫妻关系名存实亡，再继续维持夫妻名义已没有意义，故诉至法院，请求判令：1.原告与被告离婚；2.夫妻共同财产依法分割（包括位于镇海区骆驼街道某某小区住房一套，价值约105.07万元；北京现代轿车一辆，现价值约7万元）；3.婚生子郑某某甲由被告抚养，婚生女郑某某乙由原告抚养。</p><p>被告答辩称：被告不同意离婚。1.原、被告确经人介绍认识，但都是成年人，两人结婚都是自愿的；2.做手术生孩子是双方经过商量的，生下之后双方都很开心，孩子是双方感情的结晶，说明原、被告双方感情还是不错的，如果原告不同意生孩子，被告也不可能生下这两个孩子，原告所说碍于亲情生育孩子不是事实；3.原、被告结婚多年，被告尽心为家，遇到什么事都为家里着想，不像原告所说有诸多矛盾。买车的时候是由于原告工作需要，被告并不需要，但是被告仍极力支持，并且被告用自己的积蓄4万元，又借了4万元，一共8万元给原告买车，车买来后都是原告在使用。买房时，被告也同意，并没有争吵。在买房过程中，原告说要10万元，被告也向父亲和姐姐借钱用来买房。被告在经济方面都是有所支持的。对于原告提出离婚，被告觉得很突然，是原告无理取闹，没有经过慎重的考虑。至于原告提出离婚是否有其他原因被告不清楚，但对于原告在起诉状中提出的问题被告认为都不是事实。</br>原告向本院提交以下证据：</br>1．结婚证一份，欲证明原、被告系夫妻关系。被告对该份证据无异议。本院对该份证据予以认定。</br>2．出生医学证明二份，欲证明原、被告于2012年10月16日生育婚生女郑某某乙、婚生子郑某某甲的事实。被告对该组证据无异议。本院对该组证据予以认定。</br>3．居民户口簿复印件一份，欲证明原、被告的家庭关系。被告对该份证据无异议。本院对该份证据予以认定。</br>4．商品房买卖合同二份，欲证明原、被告婚后购置位于宁波市镇海区骆驼街道某某小区商品房一套。被告对该组证据无异议。本院对该组证据予以认定。</br>5．销售不动产统一发票三份，欲证明购买房屋、车库的首付金额及贷款数额。被告对该组证据的真实性无异议，但认为从该组证据中看不出贷款的具体数额。本院对该组证据的真实性予以认定。</br>6．机动车行驶证复印件一份，欲证明原、被告婚姻存续期间购买北京现代轿车（车牌号码为浙B×××××）一辆的事实。被告对该份证据无异议。本院对该份证据予以认定。</p><p>7．历史交易明细打印件二份，欲证明原告从2013年5月开始自己还某某小区商品房的贷款。被告认为该组证据的真实性无法确定。本院认为，该组证据由原告提供，仅为打印件，且被告不予认可，故本院对该份证据不予认定。</br>被告向本院提交以下证据：</br>1．房产买卖合同复印件一份，欲证明原告将原有住房出售所得收入的事实。原告对该份证据的真实性无异议。本院对该份证据的真实性予以认定。</br>2．产权部分转让换证复印件一份，欲证明镇海后葱园弄的老房屋的买入时间为2005年10月27日。原告对该份证据的真实性无异议，但称这是做房产证的时间，实际在2004年就买了房屋的一部分产权。本院对该份证据的真实性予以认定。</br>3．零售贷款历史还款交易查询打印件三份、住房公积金支取审核表复印件一份，欲证明被告共同参与后葱园弄老房子还贷款的事实。原告对该组证据无异议，但称提取公积金的钱93000元是打到被告的卡里，还贷用了65000元，剩余28000元还在被告卡里。本院对该组证据予以认定。</br>4．机动车销售统一发票、税收通用完税证、汽车装璜费发票、浙江省公路养路费票据、浙江省政府非税收入统一票据、中国工商银行储蓄存款利息清单、中国建设银行取款凭条复印件各一份，保险业专用发票复印件二份，中国工商银行个人业务凭证（专用）复印件三份，欲证明婚后所购置的车辆费用大部分由被告支付（一共花费13万多，被告承担8万元）。原告对该组证据无异议，但认为该组证据无法证明被告承担了其中的8万元。本院对该组证据的真实性予以认定。</br>5．住房公积金职工个人账户余额查询打印件一份、收入证明复印件一份、存款回单二份、存款余额信息打印件三张，欲证明原告收入、公积金余额及在部分银行存款的事实。原告对该组证据真实性无异议，但称存款回单上的款项是将后葱园弄老房子卖掉的所得的钱。本院对该组证据的真实性予以认定。</br>6．借条复印件及借条各一份，欲证明被告为了买汽车和新房子向其父亲借款的事实。原告认为该组证据的真实性无法确定，但称其对于被告借款用于买房、买车的事实予以确认。对原告确认的事实本院予以认定，但本院对该组证据不予认定。</br>7．浙江大学医学院附属妇产科医院24小时入出院记录、宁波市妇女儿童医院出院记录各一份，化验费、治疗费收费票据二份，住院收费收据五份，门诊收费收据九份，欲证明被告为生孩子就医经过及所花医药费的事实。原告对该组证据真实性无异议，称这些费用虽确实是存在的，但是不能证明这些费用都是由被告一个人承担的。本院对该组证据的真实性予以认定。</br>8.借条一份、短信打印件一份，欲证明为了买新房子，被告向其姐姐借款45000元的事实。原告对借条的真实性无法确定，称其不清楚被告的钱的来源，对短信的真实性无异议，但认为该组证据无法证明被告要证明的内容。因借条涉及案外人利益，与本案无关，故本院不予认定；因原告对短信的内容真实性无异议，故本院对其真实性予以认定。</br>经被告申请，本院向宁波银行镇海支行调取交易明细账五张，向中国工商银行镇海支行调取账户历史明细清单三张，向中国银行镇海支行调取存款历史交易明细清单二张，向交通银行镇海支行调取对私账户交易明细一张。被告以该组证据欲证明原告名下存款的去向，并要求被告对中国工商银行镇海支行的账户历史明细清单中2013年4月1日、4月26日的两笔款项的去向进行解释。原告对该组证据无异议，并解释称2013年4月1日取的45万，是因为后葱园路的房屋卖掉之后将其中的一部分钱还给原告父母；2013年4月26日存的8万元，是因为从5月份开始要还骆城芳洲房子的贷款，就从原告父母处借了8万元去还贷款。本院对上述证据予以认定。</br>经审理查明：原告郑某、被告凌某于2007年1月15日登记结婚，现原告郑某以夫妻感情破裂为由诉至法院，要求与被告凌某离婚，并依法分割夫妻共同财产，婚生子郑某某甲由被告抚养，婚生女郑某某乙由原告抚养。</br>本院认为：婚姻关系的存续应以夫妻感情为基础，离婚应以夫妻双方感情确已破裂为前提。夫妻双方应当相互信任、相互珍惜，以诚相待。原、被告虽经人介绍相识，但双方自愿结婚，至今已有七年，有一定的感情基础，且在2012年通过试管婴儿技术生下双胞胎儿女，原、被告更应珍惜这一来之不易的幸福。只要原告放弃离婚念头，多为家庭和一双儿女的共同利益着想，增强自己作为丈夫和父亲的责任感，多和被告沟通交流，夫妻和好还是有可能的。现原告没有证据证明原、被告双方的夫妻感情确已破裂，且被告也不同意离婚，故本院对原告离婚的诉讼请求不予支持。而被告也应当对自己的言行进行反思和总结，如有不当之处应予以改正，给原告更多的家庭温暖和关爱，努力改善夫妻关系。另外，原告关于分割夫妻共同财产和婚生子女抚养问题的诉讼请求，应以离婚为前提，现本院对原告离婚的诉请不予支持，对其他的诉请亦应予以驳回。据此，依照《中华人民共和国婚姻法》第三十二条，《中华人民共和国民事诉讼法》第六十四条，判决如下：</p><p>驳回原告郑某的诉讼请求。</br>本案案件受理费300元，减半收取150元，由原告郑某负担。（已预交）</br>如不服本判决，可在判决书送达之日起十五日内向本院递交上诉状，并按对方当事人的人数提出副本，上诉于浙江省宁波市中级人民法院。上诉人在收到本院送达的上诉案件受理费缴纳通知书后七日内，凭判决书到浙江省宁波市中级人民法院立案大厅收费窗口预交上诉案件受理费，如银行汇款，收款人为宁波市财政局非税资金专户，帐号：376658348992，开户银行：宁波市中国银行营业部。如邮政汇款，收款人为宁波市中级人民法院立案室。汇款时一律注明原审案号。逾期不交，作自动放弃上诉处理。</p><p>代理审判员贾毅飞</p><p>二〇一四年一月二十日</p><p>代书记员邵璐</p>'},
            {'title': '刘某与宋某离婚纠纷一审民事判决书',
             'anyou': '离婚纠纷',
             'province': '山东省',
             'court': '烟台市经济技术开发区人民法院',
             'date': '2014-02-13',
             'content': '<p align="center">烟台市经济技术开发区人民法院</p><p align="center">民事判决书</p><p align="right">（2014）开民一初字第21号</p><p>原告刘某，女，1978年12月30日生，汉族，无业。</br>被告宋某，男，1983年9月29日生，汉族，售楼员。</p><p>原告刘某诉被告宋某离婚纠纷一案，本院受理后依法由审判员纪法院独任审判，公开开庭进行了审理。原告刘某、被告宋某到庭参加诉讼。本案现已审理终结。</p><p>原告刘某诉称，原、被告双方于2013年9月17日登记结婚，婚后被告多次以种种理由向原告索要钱物，并使用原告姓名的信用卡透支消费，前后共计六万余元。自2013年10月起，双方不再共同生活，期间未生育子女，也未形成夫妻共同财产。原告认为被告的行为已经导致双方感情破裂，无法再继续生活，故请求法院依法判决双方离婚。</p><p>被告宋某辩称，原告所述不属实。我不同意原告提出的离婚请求，因为我们认识一年多了，双方是有感情基础的。且我们双方都是第二次结婚，更应该珍惜这段感情。</p><p>经审理查明，原、被告双方在互联网上相识并开始恋爱，2013年9月17日双方在烟台经济技术开发区婚姻登记处登记结婚，双方均系再婚。婚后共同生活期间，因性格不合，双方逐渐因一些家庭琐事产生矛盾，并于2013年10月份开始分居。原告以夫妻感情破裂为由诉来我院，要求与被告离婚。</br>庭审中，原告称夫妻感情已经破裂，无法与被告继续生活下去；而被告则称夫妻感情挺好，虽然双方也有矛盾，但并未影响夫妻感情。对其主张的夫妻感情已经破裂，原告未向本院提供证据加以证实。关于原告所提被告用其信用卡透支消费六万余元，被告方予以否认，原告方亦未提供相关证据加以证实。</br>上述事实，有原告提供的婚姻登记记录证明及双方当事人当庭陈述等在案为凭，足以认定。</p><p>本院认为，夫妻共同生活期间，因家庭琐事引发的冲突、矛盾在所难免，对此双方应本着互谅、互让的态度来处理，以更好地维护夫妻感情、家庭和睦。本案中原、被告双方均系再婚，均曾有过一段不幸的婚姻，双方应该更加珍惜彼此之间的感情。原、被告双方虽然已经分居，但至今分居不满两年，不足以认定夫妻感情已经破裂，故原告请求离婚，本院不予支持。希望原、被告双方在今后的共同生活中能够相互尊重、相互体贴，让夫妻关系、家庭关系变得越来越好。依照《中华人民共和国婚姻法》第三十二条之规定，判决如下：</p><p>不准原告刘某与被告宋某离婚。</br>案件受理费300元减半收取150元，由原告刘某负担。</br>如不服本判决，可在判决书送达之日起十五日内向本院递交上诉状，并按对方当事人的人数提出副本，上诉于山东省烟台市中级人民法院。</p><p>审判员纪法院</p><p>二〇一四年二月十三日</p><p>书记员王琳</p>'}]


def case_content_search_by_serials(serials, law_name, clause):
    result = []
    serials = serials.split('|')
    body = {
        "size": len(serials),
        "query": {
            "terms": {"serial": serials}
        },
        "_source": ['serial', 'title', 'province', 'court', 'anyou', 'date', 'content'],
    }
    res = es.search(index=CASE_ES_IDX_NAME, body=body)  # res = es.search(index="test-index", body={"query": {"match_all": {}}})
    for hit in res['hits']['hits']:
        result.append({'serial': hit['_source']['serial'],
                       'title': hit['_source']['title'],
                       'anyou': hit['_source']['anyou'],
                       'province': hit['_source']['province'],
                       'court': hit['_source']['court'],
                       'date': hit['_source']['date'],
                       'content': _keyword_mark(_keyword_mark(hit['_source']['content'], law_name), clause, True)})
    result = sorted(result, key=lambda x: x['date'], reverse=True)
    return result


def case_content_search_by_serials_test(serial, law_name, clause):
    return [{'title': '郑某与凌某离婚纠纷一审民事判决书',
             'anyou': '离婚纠纷',
             'province': '浙江省',
             'court': '宁波市镇海区人民法院',
             'date': '2014-01-20',
             'content': '<p align="center">宁波市镇海区人民法院</p><p align="center">民事判决书</p><p align="right">（2014）甬镇民初字第44号</p><p>原告：郑某。</br>委托代理人：杨惠康。</br>被告：凌某。</br>委托代理人：徐萍。</p><p>原告郑某与被告凌某离婚纠纷一案，本院于2013年12月26日立案受理后，依法由代理审判员贾毅飞适用简易程序独任审判，于2014年1月16日公开开庭进行了审理。原告郑某及其委托代理人杨惠康、被告凌某及其委托代理人徐萍到庭参加诉讼。本案现已审理终结。</p><p>原告郑某起诉称：原告于2006年经长辈介绍与被告认识，见过数面后，原告就向长辈提及两人不合适，包括生活习惯、消费观念以及业余爱好等。但因双方父亲曾经是同学，所以极力撮合此事，原告受长辈劝解的影响，最终接受，于2007年1月15日到镇海区招宝山街道领取结婚证。由于婚前双方缺乏了解，感情基础薄弱，沟通困难，婚后两人因各种原因时常有大小争吵。主要原因有：1.夫妻生活不正常。婚后夫妻生活不正常，一直未生育，碍于亲情压力，2012年1月，在杭州妇儿医院接受试管婴儿手术，同年10月16日，诞下双胞胎，儿子郑某某甲，女儿郑某某乙。但是孩子的到来反而突显了双方生活习惯、教育理念的矛盾，由此引发的争吵越来越频繁。2.经济问题。结婚以来双方经济都是独立的，家里的生活支出基本都是AA制。遇到买车、买房等大件支出，总有无休止的争吵、矛盾。3.生活习惯、业余爱好差异大的问题。因被告在安徽巢湖出生，一直到毕业后才来到宁波工作。所以双方在生活习惯上差异较大，而且对于孩子的教育观念也很不一样，由此经常引发大吵。而双方因性格、观念差异，沟通方式匮乏，很难达成一致。以上种种原因，导致争吵不断，怨恨日渐加重。由于无法忍受这种没有感情、没有温暖的夫妻生活，原、被告曾于2013年5月到民政局协议离婚，被告要求抚养两个子女，而原告认为被告在镇海区骆驼街道上班，其父母住址在北仑区，住宅面积约60平方，父母医疗等关系尚在安徽巢湖，需要定期返回核销医疗费用，存在很多不便因素，所以原告认为被告抚养两个孩子有困难，不利于孩子更好地成长。而原告在镇海上班，父母住在镇海繁景小区，住宅面积约90平方，附近生活、教育等配套设施成熟，利于孩子成长，所以原告希望双方各自抚养一个。因双方在孩子抚养权方面无法达成一致，所以无法成功协议离婚。原告也于2013年5月开始一直住在父母家，被告租住在骆驼，双方已是分居生活。期间双方家长、亲戚也曾试图调解，在详细了解情况后也不再提出反对意见。因原、被告双方没有感情基础，婚后夫妻感情一直未能建立，而且呈进一步恶化趋势，现夫妻感情完全、彻底破裂，夫妻关系名存实亡，再继续维持夫妻名义已没有意义，故诉至法院，请求判令：1.原告与被告离婚；2.夫妻共同财产依法分割（包括位于镇海区骆驼街道某某小区住房一套，价值约105.07万元；北京现代轿车一辆，现价值约7万元）；3.婚生子郑某某甲由被告抚养，婚生女郑某某乙由原告抚养。</p><p>被告答辩称：被告不同意离婚。1.原、被告确经人介绍认识，但都是成年人，两人结婚都是自愿的；2.做手术生孩子是双方经过商量的，生下之后双方都很开心，孩子是双方感情的结晶，说明原、被告双方感情还是不错的，如果原告不同意生孩子，被告也不可能生下这两个孩子，原告所说碍于亲情生育孩子不是事实；3.原、被告结婚多年，被告尽心为家，遇到什么事都为家里着想，不像原告所说有诸多矛盾。买车的时候是由于原告工作需要，被告并不需要，但是被告仍极力支持，并且被告用自己的积蓄4万元，又借了4万元，一共8万元给原告买车，车买来后都是原告在使用。买房时，被告也同意，并没有争吵。在买房过程中，原告说要10万元，被告也向父亲和姐姐借钱用来买房。被告在经济方面都是有所支持的。对于原告提出离婚，被告觉得很突然，是原告无理取闹，没有经过慎重的考虑。至于原告提出离婚是否有其他原因被告不清楚，但对于原告在起诉状中提出的问题被告认为都不是事实。</br>原告向本院提交以下证据：</br>1．结婚证一份，欲证明原、被告系夫妻关系。被告对该份证据无异议。本院对该份证据予以认定。</br>2．出生医学证明二份，欲证明原、被告于2012年10月16日生育婚生女郑某某乙、婚生子郑某某甲的事实。被告对该组证据无异议。本院对该组证据予以认定。</br>3．居民户口簿复印件一份，欲证明原、被告的家庭关系。被告对该份证据无异议。本院对该份证据予以认定。</br>4．商品房买卖合同二份，欲证明原、被告婚后购置位于宁波市镇海区骆驼街道某某小区商品房一套。被告对该组证据无异议。本院对该组证据予以认定。</br>5．销售不动产统一发票三份，欲证明购买房屋、车库的首付金额及贷款数额。被告对该组证据的真实性无异议，但认为从该组证据中看不出贷款的具体数额。本院对该组证据的真实性予以认定。</br>6．机动车行驶证复印件一份，欲证明原、被告婚姻存续期间购买北京现代轿车（车牌号码为浙B×××××）一辆的事实。被告对该份证据无异议。本院对该份证据予以认定。</p><p>7．历史交易明细打印件二份，欲证明原告从2013年5月开始自己还某某小区商品房的贷款。被告认为该组证据的真实性无法确定。本院认为，该组证据由原告提供，仅为打印件，且被告不予认可，故本院对该份证据不予认定。</br>被告向本院提交以下证据：</br>1．房产买卖合同复印件一份，欲证明原告将原有住房出售所得收入的事实。原告对该份证据的真实性无异议。本院对该份证据的真实性予以认定。</br>2．产权部分转让换证复印件一份，欲证明镇海后葱园弄的老房屋的买入时间为2005年10月27日。原告对该份证据的真实性无异议，但称这是做房产证的时间，实际在2004年就买了房屋的一部分产权。本院对该份证据的真实性予以认定。</br>3．零售贷款历史还款交易查询打印件三份、住房公积金支取审核表复印件一份，欲证明被告共同参与后葱园弄老房子还贷款的事实。原告对该组证据无异议，但称提取公积金的钱93000元是打到被告的卡里，还贷用了65000元，剩余28000元还在被告卡里。本院对该组证据予以认定。</br>4．机动车销售统一发票、税收通用完税证、汽车装璜费发票、浙江省公路养路费票据、浙江省政府非税收入统一票据、中国工商银行储蓄存款利息清单、中国建设银行取款凭条复印件各一份，保险业专用发票复印件二份，中国工商银行个人业务凭证（专用）复印件三份，欲证明婚后所购置的车辆费用大部分由被告支付（一共花费13万多，被告承担8万元）。原告对该组证据无异议，但认为该组证据无法证明被告承担了其中的8万元。本院对该组证据的真实性予以认定。</br>5．住房公积金职工个人账户余额查询打印件一份、收入证明复印件一份、存款回单二份、存款余额信息打印件三张，欲证明原告收入、公积金余额及在部分银行存款的事实。原告对该组证据真实性无异议，但称存款回单上的款项是将后葱园弄老房子卖掉的所得的钱。本院对该组证据的真实性予以认定。</br>6．借条复印件及借条各一份，欲证明被告为了买汽车和新房子向其父亲借款的事实。原告认为该组证据的真实性无法确定，但称其对于被告借款用于买房、买车的事实予以确认。对原告确认的事实本院予以认定，但本院对该组证据不予认定。</br>7．浙江大学医学院附属妇产科医院24小时入出院记录、宁波市妇女儿童医院出院记录各一份，化验费、治疗费收费票据二份，住院收费收据五份，门诊收费收据九份，欲证明被告为生孩子就医经过及所花医药费的事实。原告对该组证据真实性无异议，称这些费用虽确实是存在的，但是不能证明这些费用都是由被告一个人承担的。本院对该组证据的真实性予以认定。</br>8.借条一份、短信打印件一份，欲证明为了买新房子，被告向其姐姐借款45000元的事实。原告对借条的真实性无法确定，称其不清楚被告的钱的来源，对短信的真实性无异议，但认为该组证据无法证明被告要证明的内容。因借条涉及案外人利益，与本案无关，故本院不予认定；因原告对短信的内容真实性无异议，故本院对其真实性予以认定。</br>经被告申请，本院向宁波银行镇海支行调取交易明细账五张，向中国工商银行镇海支行调取账户历史明细清单三张，向中国银行镇海支行调取存款历史交易明细清单二张，向交通银行镇海支行调取对私账户交易明细一张。被告以该组证据欲证明原告名下存款的去向，并要求被告对中国工商银行镇海支行的账户历史明细清单中2013年4月1日、4月26日的两笔款项的去向进行解释。原告对该组证据无异议，并解释称2013年4月1日取的45万，是因为后葱园路的房屋卖掉之后将其中的一部分钱还给原告父母；2013年4月26日存的8万元，是因为从5月份开始要还骆城芳洲房子的贷款，就从原告父母处借了8万元去还贷款。本院对上述证据予以认定。</br>经审理查明：原告郑某、被告凌某于2007年1月15日登记结婚，现原告郑某以夫妻感情破裂为由诉至法院，要求与被告凌某离婚，并依法分割夫妻共同财产，婚生子郑某某甲由被告抚养，婚生女郑某某乙由原告抚养。</br>本院认为：婚姻关系的存续应以夫妻感情为基础，离婚应以夫妻双方感情确已破裂为前提。夫妻双方应当相互信任、相互珍惜，以诚相待。原、被告虽经人介绍相识，但双方自愿结婚，至今已有七年，有一定的感情基础，且在2012年通过试管婴儿技术生下双胞胎儿女，原、被告更应珍惜这一来之不易的幸福。只要原告放弃离婚念头，多为家庭和一双儿女的共同利益着想，增强自己作为丈夫和父亲的责任感，多和被告沟通交流，夫妻和好还是有可能的。现原告没有证据证明原、被告双方的夫妻感情确已破裂，且被告也不同意离婚，故本院对原告离婚的诉讼请求不予支持。而被告也应当对自己的言行进行反思和总结，如有不当之处应予以改正，给原告更多的家庭温暖和关爱，努力改善夫妻关系。另外，原告关于分割夫妻共同财产和婚生子女抚养问题的诉讼请求，应以离婚为前提，现本院对原告离婚的诉请不予支持，对其他的诉请亦应予以驳回。据此，依照《中华人民共和国婚姻法》第三十二条，《中华人民共和国民事诉讼法》第六十四条，判决如下：</p><p>驳回原告郑某的诉讼请求。</br>本案案件受理费300元，减半收取150元，由原告郑某负担。（已预交）</br>如不服本判决，可在判决书送达之日起十五日内向本院递交上诉状，并按对方当事人的人数提出副本，上诉于浙江省宁波市中级人民法院。上诉人在收到本院送达的上诉案件受理费缴纳通知书后七日内，凭判决书到浙江省宁波市中级人民法院立案大厅收费窗口预交上诉案件受理费，如银行汇款，收款人为宁波市财政局非税资金专户，帐号：376658348992，开户银行：宁波市中国银行营业部。如邮政汇款，收款人为宁波市中级人民法院立案室。汇款时一律注明原审案号。逾期不交，作自动放弃上诉处理。</p><p>代理审判员贾毅飞</p><p>二〇一四年一月二十日</p><p>代书记员邵璐</p>'},
            {'title': '刘某与宋某离婚纠纷一审民事判决书',
             'anyou': '离婚纠纷',
             'province': '山东省',
             'court': '烟台市经济技术开发区人民法院',
             'date': '2014-02-13',
             'content': '<p align="center">烟台市经济技术开发区人民法院</p><p align="center">民事判决书</p><p align="right">（2014）开民一初字第21号</p><p>原告刘某，女，1978年12月30日生，汉族，无业。</br>被告宋某，男，1983年9月29日生，汉族，售楼员。</p><p>原告刘某诉被告宋某离婚纠纷一案，本院受理后依法由审判员纪法院独任审判，公开开庭进行了审理。原告刘某、被告宋某到庭参加诉讼。本案现已审理终结。</p><p>原告刘某诉称，原、被告双方于2013年9月17日登记结婚，婚后被告多次以种种理由向原告索要钱物，并使用原告姓名的信用卡透支消费，前后共计六万余元。自2013年10月起，双方不再共同生活，期间未生育子女，也未形成夫妻共同财产。原告认为被告的行为已经导致双方感情破裂，无法再继续生活，故请求法院依法判决双方离婚。</p><p>被告宋某辩称，原告所述不属实。我不同意原告提出的离婚请求，因为我们认识一年多了，双方是有感情基础的。且我们双方都是第二次结婚，更应该珍惜这段感情。</p><p>经审理查明，原、被告双方在互联网上相识并开始恋爱，2013年9月17日双方在烟台经济技术开发区婚姻登记处登记结婚，双方均系再婚。婚后共同生活期间，因性格不合，双方逐渐因一些家庭琐事产生矛盾，并于2013年10月份开始分居。原告以夫妻感情破裂为由诉来我院，要求与被告离婚。</br>庭审中，原告称夫妻感情已经破裂，无法与被告继续生活下去；而被告则称夫妻感情挺好，虽然双方也有矛盾，但并未影响夫妻感情。对其主张的夫妻感情已经破裂，原告未向本院提供证据加以证实。关于原告所提被告用其信用卡透支消费六万余元，被告方予以否认，原告方亦未提供相关证据加以证实。</br>上述事实，有原告提供的婚姻登记记录证明及双方当事人当庭陈述等在案为凭，足以认定。</p><p>本院认为，夫妻共同生活期间，因家庭琐事引发的冲突、矛盾在所难免，对此双方应本着互谅、互让的态度来处理，以更好地维护夫妻感情、家庭和睦。本案中原、被告双方均系再婚，均曾有过一段不幸的婚姻，双方应该更加珍惜彼此之间的感情。原、被告双方虽然已经分居，但至今分居不满两年，不足以认定夫妻感情已经破裂，故原告请求离婚，本院不予支持。希望原、被告双方在今后的共同生活中能够相互尊重、相互体贴，让夫妻关系、家庭关系变得越来越好。依照《中华人民共和国婚姻法》第三十二条之规定，判决如下：</p><p>不准原告刘某与被告宋某离婚。</br>案件受理费300元减半收取150元，由原告刘某负担。</br>如不服本判决，可在判决书送达之日起十五日内向本院递交上诉状，并按对方当事人的人数提出副本，上诉于山东省烟台市中级人民法院。</p><p>审判员纪法院</p><p>二〇一四年二月十三日</p><p>书记员王琳</p>'}]


def case_content_download_by_serials(serials):
    result = []
    body = {
        "size": len(serials),
        "query": {
            "terms": {"serial": serials}
        },
        "_source": ['serial', 'anyou', 'sucheng', 'chaming', 'renwei', 'panjue', 'content'],
    }
    res = es.search(index=CASE_ES_IDX_NAME, body=body)
    for hit in res['hits']['hits']:
        result.append([hit['_source']['serial'],
                       hit['_source']['anyou'],
                       hit['_source']['sucheng'],
                       hit['_source']['chaming'],
                       hit['_source']['renwei'],
                       hit['_source']['panjue'],
                       hit['_source']['content'].replace('<p>', '').replace('</p>', '\n')])
    result = pd.DataFrame(result, columns=['serial','reference','litigant','fact','viewpoint','result','content'])
    result = result.sort_values(by=['reference'])
    filename = 'case' + datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S') + '.csv'
    result.to_csv('./download/' + filename, index=False, encoding='utf-8')
    return filename


if __name__=='__main__':
    print(_keyword_mark('劳动的报酬', '劳动报酬'))
