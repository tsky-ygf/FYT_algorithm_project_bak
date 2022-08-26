import os
import json
import sys

import tqdm

import pandas as pd
from math import ceil
from sklearn.model_selection import train_test_split


def process_json(path_input, path_output):
    train = os.path.join(path_output, "train.json")  # 合并后的文件名train_num10.json
    dev = os.path.join(path_output, "dev.json")  # 合并后的文件名dev_num10.json
    test = os.path.join(path_output, "test.json")  # 合并后的文件名dev_num10.json

    with open(train, "w", encoding="utf-8") as f60, open(dev, "w", encoding="utf-8") as f20d, open(test, "w",
                                                                                                   encoding="utf-8") as f20t:
        for file in os.listdir(path_input):
            item_num, item_temp = 0, 0
            item_num = len(open(os.path.join(path_input, file), 'r', encoding="utf-8").readlines())
            with open(os.path.join(path_input, file), 'r', encoding='utf-8') as f:
                for item in f:
                    item = json.loads(item.strip())
                    item_temp += 1
                    if item_temp < item_num * 0.6:
                        js_con60 = json.dumps(''.join(json.dumps(item, ensure_ascii=False)), ensure_ascii=False)
                        f60.write(eval(js_con60) + '\n')
                    elif item_num * 0.6 <= item_temp < item_num * 0.8:
                        js_con20d = json.dumps(''.join(json.dumps(item, ensure_ascii=False)), ensure_ascii=False)
                        f20d.write(eval(js_con20d) + '\n')
                    else:
                        js_con20t = json.dumps(''.join(json.dumps(item, ensure_ascii=False)), ensure_ascii=False)
                        f20t.write(eval(js_con20t) + '\n')
            f.close()
        f60.close()
        f20d.close()
        f20t.close()


def split_with_sklearn(path_input, path_output):
    train = os.path.join(path_output, "train.json")  # 合并后的文件名train_num10.json
    dev = os.path.join(path_output, "dev.json")  # 合并后的文件名dev_num10.json
    test = os.path.join(path_output, "test.json")  # 合并后的文件名dev_num10.json

    for file in os.listdir(path_input):
        with open(os.path.join(path_input, file), 'r', encoding='utf-8') as f:
            all_item_list = json.load(f)
            item_list = all_item_list['RECORDS']
            df = pd.DataFrame(item_list)
            print(df['factor'].values.tolist())
            spec_train, spec_test = train_test_split(df, test_size=0.2, stratify=df['factor'].values.tolist())
            spec_train.to_json(train, orient='records', force_ascii=False)
            spec_test.to_json(dev, orient='records', force_ascii=False)

    return 0


def ensemble2split(path_input, path_output):
    train = os.path.join(path_output, "train.json")  # 合并后的文件名train_num10.json
    dev = os.path.join(path_output, "dev.json")  # 合并后的文件名dev_num10.json
    test = os.path.join(path_output, "test.json")  # 合并后的文件名dev_num10.json

    for file in os.listdir(path_input):
        with open(os.path.join(path_input, file), 'r', encoding='utf-8') as f:
            item_to_split = []
            for item in f:
                item = json.loads(item.strip())
                item["situa_label"] = item[0]["situation"]
                item_to_split.append(item)
            # df = pd.DataFrame(item_list)
            # print(df['factor'].values.tolist())
            # spec_train, spec_test = train_test_split(df, test_size=0.2, stratify=df['factor'].values.tolist())
            # spec_train.to_json(train, orient='records', force_ascii=False)
            # spec_test.to_json(dev, orient='records', force_ascii=False)

            label_sit_spe = []
            for item_split_label in item_to_split:
                label_situa = item_split_label[0]["situation"]
                if label_situa not in label_sit_spe:
                    label_sit_spe.append(label_situa)
            data_situa_list = [[] for j in range(len(label_sit_spe))]
            for item_split_words in item_to_split:
                label_situa = item_split_label[0]["situation"]
                if label_situa in label_sit_spe:
                    data_situa_list[label_sit_spe.index(label_situa)].append(item_split_words)
            augment_list = []
            copy_list = []
            split_list = []
            for item_data_sit in data_situa_list:
                if len(item_data_sit) < 33:
                    augment_list.append(item_data_sit)
                elif 33 < len(item_data_sit) < 133:
                    copy_list.append(item_data_sit)
                else:
                    split_list.append(item_data_sit)
            min_len = sys.maxsize
            for item_split in split_list:
                if len(item_split) < min_len:
                    min_len = len(item_split)
            split_num = 0
            if min_len % 100 > 33:
                split_num = ceil(min_len / 100)
            else:
                split_num = int(min_len / 100)
            for i in range(split_num):
                final_list_temp = []
                data_train_path = 'LawEntityExtraction/data/bert/spec_train_ner' + str(i) + '.txt'
                data_dev_path = 'LawEntityExtraction/data/bert/spec_dev_ner' + str(i) + '.txt'
                for item_copy_list in copy_list:
                    for sub_copy_list in item_copy_list:
                        final_list_temp.append(sub_copy_list)
                for item_split_list in split_list:
                    for sub_split_list in item_split_list[i * 100:(i + 1) * 100]:
                        final_list_temp.append(sub_split_list)
                finalList = []


if __name__ == '__main__':
    path_input, path_output = "LawEntityExtraction/data/spec_situa_origin/spec_situation_2.json", "LawEntityExtraction/data/bert "

    if not os.path.exists(path_output):
        os.mkdir(path_output)
    # process_json(path_input, path_output)
    # split_with_sklearn(path_input, path_output)
    ensemble2split(path_input, path_output)
