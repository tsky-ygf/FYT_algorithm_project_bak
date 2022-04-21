# -*- coding: utf-8 -*-
import re

##############################################################################################################################################
#
# 裁判文书结构化信息提取
#
##############################################################################################################################################


def html_clean(html_string):
    html_string = html_string.replace('(', '（').replace(')', '）').replace(',', '，').replace(':', '：').replace(';', '；').replace('?', '？').replace('!', '！')
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
    '(本院.{0,5}(查明|查清|认定)如下)',
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


def chaming_extract(html_string):
    html_string = html_clean(html_string)
    informations = re.findall('<p.*?</p>', html_string)
    html_string = ''.join(informations[3:])
    html_string = re.sub('[^。！？：]</br>', lambda x: x.group(0)[0], html_string)

    chaming = None
    for p in chaming_pattern:
        if len(re.findall(p + '.{5,}', html_string)) > 0:
            pattern = re.findall(p, html_string)[0][0]
            chaming = pattern_extract(html_string, pattern)
            for sentence in end_sentences:
                if len(re.findall(sentence, chaming)) > 0:
                    pattern = re.findall(sentence, chaming)[0][0]
                    chaming = chaming[:chaming.rindex(pattern)]
            chaming = chaming.replace('</br>', '')
            if len(re.findall('与原告.{0,3}诉称[^，。；：！？不]*一致', chaming)) > 0:
                chaming = None
            break
    return chaming


def chaming_fact_extract(x):
    if x != x or x is None:
        return None
    x = re.sub('。[一二三四五六七八九]', lambda r: '，' + r[0][1], x)
    if len(re.findall('((约定|第.{1,3}条)[^。？！]*“.*”)', x)) > 0:
        pattern = re.findall('((约定|第.{1,3}条)[^。？！]*“.*”)', x)[0][0]
        x1 = x[:x.index(pattern)]
        x2 = x[x.index(pattern)+len(pattern):]
        while len(x1)>0 and x1[-1] not in ['，', '；', '：', '。', '？', '！']:
            x1 = x1[:-1]
        x = x1 + x2

    sentences = []
    for sentence in re.split('[。？！]', x):
        if len(sentence) == 0:
            continue
        if len(re.findall('((如果|应当|应该|理应|认为|权利|约定|要求|请求|第.{1,3}条).*$)', sentence)) > 0:
            pattern = re.findall('((如果|应当|应该|理应|认为|权利|约定|要求|请求|第.{1,3}条).*$)', sentence)[0][0]
            sentence = sentence[:sentence.index(pattern)]
            while len(sentence)>0 and sentence[-1] not in ['，', '；', '：']:
                sentence = sentence[:-1]
            if len(sentence)>0 and sentence[-1] in ['，', '；', '：']:
                sentence = sentence[:-1]
        if len(sentence) > 5:
            sentences.append(sentence)
    sentences = '。'.join(sentences) + '。'
    return sentences


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


def renwei_fact_extract(x):
    if x != x or x is None:
        return None
    x = re.sub('。[一二三四五六七八九]', lambda r: '，' + r[0][1], x)
    sentences = []
    for sentence in re.split('[。？！]', x[5:]):
        if len(sentence) == 0:
            continue
        if len(re.findall('到庭|传唤|民事判决书|民事裁决书|民事调解书|仲裁裁决书|当事人|劳动者|所谓.*是指|争议', sentence)) > 0:
            continue
        if len(sentence) > 5:
            if len(re.findall('((根据|依据|《|依照|按照)[^。；？！]*规定.*$)', sentence))>0:
                pattern = re.findall('((根据|依据|《|依照|按照)[^。；？！]*规定.*$)', sentence)[0][0]
                sentence = sentence[:sentence.index(pattern)]
                while len(sentence)>0 and sentence[-1] not in ['，']:
                    sentence = sentence[:-1]
            if len(sentence)>0:
                sentences.append(sentence)
    sentences = '。'.join(sentences) + '。'

    result = ''
    for sentence in re.split('[，。；]', sentences):
        if len(re.findall('^故|因此|如果|据此|应当|应该|应予|理应|应.*支付|应.*给付|应.*赔偿|认为|认定|采信|采纳|准许|驳回|支持|成立|异议|无须|无需|根据|权利|权益|法律规定|于法不悖|法律依据', sentence)) == 0 and len(sentence) > 4:
            result += sentence + '。'
    if len(result) == 0:
        return None
    if result[0] in ['，', '。', '：', '；']:
        result = result[1:]
    return result


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
        while len(fatiao)>0 and fatiao[-1] not in ['款', '条', '项', '》']:
            fatiao = fatiao[:-1]
    return fatiao


def fatiao_correct(fatiao):
    if fatiao!=fatiao or fatiao is None:
        return fatiao

    result = []
    last_law = None
    for ft in re.split('[、，；及《]|和第|和中华|依据|参照', fatiao):
        if '第' not in ft and '条' not in ft and '》' not in ft:
            continue
        if ft.startswith('人民共和国'):
            ft = '中华' + ft
        if '》' in ft and '《' not in ft:
            ft = '《' + ft
        if '第' not in ft and '条' in ft:
            start_index = ft.index('条')
            while start_index>0 and len(re.findall('[一二三四五六七八九十百千零]', ft[start_index - 1])) > 0:
                start_index -= 1
            ft = ft[:start_index] + '第' + ft[start_index:]
        elif '第' in ft and '条' not in ft:
            start_index = ft.index('第')
            while start_index<len(ft)-1 and len(re.findall('[一二三四五六七八九十百千零]', ft[start_index + 1])) > 0:
                start_index += 1
            ft = ft[:start_index+1] + '条' + ft[start_index+1:]

        if '第' not in ft and '条' not in ft:
            result.append(ft)
        elif ft[:ft.index('第')] == '':
            if last_law is not None:
                start_index = ft.index('条')
                while ft[start_index]!='第':
                    start_index -= 1
                result.append(last_law + ft[start_index: ft.index('条') + 1])
        else:
            start_index = ft.index('条')
            while ft[start_index]!='第':
                start_index -= 1
            last_law = ft[:ft.index('第')]
            result.append(last_law + ft[start_index: ft.index('条') + 1])
    return '|'.join(result)


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
