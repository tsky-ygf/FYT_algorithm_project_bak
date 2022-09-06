import json
import requests

from Utils import print_run_time


@print_run_time
def req_suqiu(url_suqiu, input_json):
    r = requests.post(url_suqiu, json=input_json)  #  r 的格式 {'suqiu_pro': suqiu_pro, "status": 0}
                                                   # suqiu_pro 的格式 为dict 诉求和概率的键值对
    res = r.json()
    print(res)
    return res
#
#
# res['content'] = content # 重新加是上内容和用户输入的诉求
# res['suqiu'] = suqiu
@print_run_time
def req_situa(url_situation, input_json):
    r = requests.post(url_situation, json=input_json) # 预测情形 输入格式还是input_json  输出为  {'sit_pro': sit_pro, "status": 0}
    res = r.json()
    print(res)
    return res

if __name__=='__main__':
    url_suqiu = 'http://172.19.82.199:7995/suqiureview'
    url_situation = 'http://172.19.82.199:7995/situationreview'
    content = "原告付某向本院提出诉讼请求：1、请求判令原被告解除婚姻关系;2、请求判令原被告女儿魏晋新抚养权归原告所有，被告向原告按月支付抚养费3，000元至魏晋新18周岁;3、请求依法分割夫妻共同财产;4、请求判令被告承担本案诉讼费。事实和理由：原被告双方于2011年相识，系朋友关系，后双方自由恋爱，于2013年1月7日登记结婚。婚后双方感情尚可，并于2013年婚后购置闵行区春申路XXX弄XXX号XXX室房屋一套，房屋所有权登记于原被告双方名下。两人于2014年10月26日生育一女，取名魏晋新。此后，被告脾气暴躁，大男子主义思想严重，经常为一些小事与原告吵架，并多次施于家暴行为。同时不履行作为一个丈夫与父亲的应有的义务和责任，且不求上进，学业荒废，至今未毕业，也没有正式工作，给本不宽裕的家庭增添了不少负担。原告忍无可忍，已从双方婚后购置的房屋中搬离数月。由于婚前双方缺乏了解，感情基础薄弱，双方人生观和价值不同，致原本脆弱的夫妻感情完全破裂，不存在和好的可能。因此原告经与被告多次沟通无效后，现向贵院提起诉讼，望判如所请。本院认为，夫妻关系的存续应以感情为基础，夫妻感情破裂是解除婚姻的条件。本案中，原告要求与被告离婚，被告同意与原告离婚，可以说明双方感情已破裂，基于原被告双方对离婚意思表示一致，故原告要求与被告离婚之诉请，本院予以准许。关于原告诉称被告存在家暴行为，根据本案证据尚不足以认定，本院不予认可。"
    content = "想要离婚, 划分财产怎么办？"
    suqiu = "离婚,财产分割"
    input_json = {
        "content": content
        , "suqiu": suqiu  # 预测诉求时，目前输入参数无效， 预测情形时需要输入
    }
    req_suqiu(url_suqiu, input_json)
    req_situa(url_situation, input_json)

