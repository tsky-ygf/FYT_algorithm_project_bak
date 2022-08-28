import traceback
import logging
import logging.handlers
from flask import Flask, request
from flask_cors import CORS
import json

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '3'

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/situationreview', methods=["post"])
def get_situation_res():
    try:
        # input_str = request.get_data()
        # content, suqiu = input_str.split(str.encode('\t\t'))
        input_json = request.get_data()
        if input_json is not None:
            input_dict = json.loads(input_json.decode("utf-8"))
            text = input_dict['content']
            suqiu_type = get_suqiu_type(text)

            # text = content.decode("utf-8")
            # suqiu = suqiu.decode("utf-8")
            # suqiu_type = get_suqiu_type(text)

            situation_type = get_situation_type(json.loads(suqiu_type)['suqiu_type'][0], text, json.loads(suqiu_type)['probability'])
            print(text)
            print(suqiu_type)
            print(situation_type)
            return json.dumps(situation_type, ensure_ascii=False)
        else:
            return json.dumps({"error_msg": "no data", "status": 1}, ensure_ascii=False)

    except Exception as e:
        logging.info(traceback.format_exc())
        return json.dumps({"error_msg": "unknown error:" + repr(e), "status": 1}, ensure_ascii=False)


def get_split_content(input_words):
    text_merge = []

    return text_merge


def get_suqiu_type(text):
    from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

    situation = BasicBertSituation(
        config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config/config_suqiu_cls.yaml")
    suqiu_type, pro = situation.review_main(content=text, mode="text")

    return json.dumps({'suqiu_type': suqiu_type, 'probability': float(pro), "status": 0}, ensure_ascii=False)


def get_situation_type(suqiu_type, text, suqiu_pro):
    situation_res = ''

    if suqiu_type == '离婚':
        from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

        situation = BasicBertSituation(
            config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config"
                        "/config_divorce_cls.yaml")
    elif suqiu_type == '财产分割':
        from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

        situation = BasicBertSituation(
            config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config"
                        "/config_property_division.yaml")
    elif suqiu_type == '支付赡养费':  # TODO 下面有一个大类，其他为小类，使用规则

        return json.dumps({'suqiu_type': suqiu_type, 'situation': '被赡养人缺乏劳动能力或生活困难要求子女支付赡养费', 'probability': suqiu_pro, "status": 0}, ensure_ascii=False)

    elif suqiu_type == '支付抚养费':
        from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

        situation = BasicBertSituation(
            config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config"
                        "/config_pay_support.yaml")
    elif suqiu_type == '增加抚养费':
        from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

        situation = BasicBertSituation(
            config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config"
                        "/config_increase_support.yaml")
    elif suqiu_type == '行使探望权':  # TODO 下面有一个大类，其他为小类，使用规则

        return json.dumps({'suqiu_type': suqiu_type, 'situation': '不直接抚养子女的父或母，有权利探望子女', 'probability': suqiu_pro, "status": 0}, ensure_ascii=False)

    elif suqiu_type == '确认婚姻无效':
        from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

        situation = BasicBertSituation(
            config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config"
                        "/config_confirm_invalid.yaml")
    elif suqiu_type == '减少抚养费':
        from LawEntityExtraction.LabelClsReview.basic_review import BasicBertSituation

        situation = BasicBertSituation(
            config_path="/home/fyt/huangyulin/project/fyt/LawEntityExtraction/LabelClsReview/Config"
                        "/config_reduce_support.yaml")
    else:
        raise Exception("暂时不支持该诉求类型")

    res, pro = situation.review_main(content=text, mode="text")
    return json.dumps({'suqiu_type': suqiu_type, 'situation': res[0], 'probability': float(pro), "status": 0}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7999, debug=True)  # , use_reloader=False)
