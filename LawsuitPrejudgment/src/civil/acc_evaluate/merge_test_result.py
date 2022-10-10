import json
import os

import pandas as pd

des_folder = '../data/test_cases_20220210'

# 文件地址, problem, suqius, sentence, 'question_answers', result
# {"problem": "婚姻家庭", "suqius": "确认婚姻无效", "sentence": "xxx",
# "question_answers": [{"question": "是否存在以下情形？:一方患有禁止结婚的疾病;一方未到法定婚龄;双方属于三代以内旁系血亲;双方属于直系血亲;双方非自愿结婚;一方重婚;以上都没有", "answers": "一方患有禁止结婚的疾病"},
# {"question": "病情是否久治不愈？:是;否", "answers": "否"}],
# "result": [{"suqiu": "确认婚姻无效", "support_or_not": "不支持", "possibility_support": 0.308, "true_label": "0"}]}
def merge_folder(folder):
    records = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder, file_name)
            data = json.load(open(file_path))
            if data['result']:
                records.append([file_path, data['problem'], data['suqius'], data['sentence'], data['question_answers'],
                                data['result'], data['result_code'], data['label_code']])

    df = pd.DataFrame(records, columns=['id', 'problem', 'suqius', 'sentence', 'question_answers', 'result',
                                        'result_code', 'label_code'])
    df.to_excel(os.path.join(folder, 'all_20220212.xlsx'))


if __name__ == '__main__':
    merge_folder(des_folder)
