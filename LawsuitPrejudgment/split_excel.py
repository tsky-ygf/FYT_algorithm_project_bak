import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd

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

def sample_from_excel(sample_list, folder, des_path):
    id_list = defaultdict(list)
    for fname in os.listdir(folder):
        file_path = os.path.join(folder, fname)
        df = pd.read_excel(file_path, keep_default_na=True, dtype={'f7': str})
        df = df[df['f10'] == '判决']
        for jiufen_type in sample_list.keys():
            tmp_df = df[df['f12'] == jiufen_type]
            tmp_list = tmp_df['id'].values.tolist()
            id_list[jiufen_type].extend(tmp_list)
        print(', '.join(['{}:{}'.format(k, len(v)) for k, v in sorted(id_list.items())]), fname)

    np.random.seed(3)
    id_list_set_train = set()
    id_list_set_test = set()
    total_id = []
    total_jiufen = []
    for k, v in sorted(id_list.items()):
        v = sorted(v)
        id_list_selected = np.random.choice(v, sum(sample_list[k]), False)
        id_list_set_train = id_list_set_train | set(id_list_selected[:sample_list[k][0]])
        id_list_set_test = id_list_set_test | set(id_list_selected[sample_list[k][0]:])
        total_id.extend(v)
        total_jiufen.extend([k]*len(v))

    df = pd.DataFrame({'id': total_id, '纠纷类型': total_jiufen})
    def select_train_test(each_id):
        if each_id in id_list_set_train:
            return 1
        elif each_id in id_list_set_test:
            return 2
        else:
            return 0
    df['sample'] = df['id'].apply(select_train_test)
    df.to_csv(des_path)

def split_train_test(folder, des_path, des_folder):
    df = pd.read_csv(des_path)
    id_list = defaultdict(dict)
    for i, row in df.iterrows():
        if row['sample'] > 0:
            if 'train' not in id_list[row['纠纷类型']]:
                id_list[row['纠纷类型']] = {'train': set(), 'test': set(), 'df_train': [], 'df_test': []}
            if row['sample'] == 1:
                id_list[row['纠纷类型']]['train'].add(row['id'])
            else:
                id_list[row['纠纷类型']]['test'].add(row['id'])

    for fname in os.listdir(folder):
        print(fname)
        file_path = os.path.join(folder, fname)
        df = pd.read_excel(file_path, keep_default_na=True, dtype={'f7': str})
        for jiufen_type, v in id_list.items():
            tmp_df = df[df['id'].apply(lambda x: x in v['train'])]
            v['df_train'].append(tmp_df)
            tmp_df = df[df['id'].apply(lambda x: x in v['test'])]
            v['df_test'].append(tmp_df)
    for jiufen_type, v in id_list.items():
        df = pd.concat(v['df_train'])
        df.to_excel(os.path.join(des_folder, jiufen_type+'_train.xlsx'), index=False)
        df = pd.concat(v['df_test'])
        df.to_excel(os.path.join(des_folder, jiufen_type + '_test.xlsx'), index=False)

def extract_paragraph(folder):
    for fname in os.listdir(folder):
        file_path = os.path.join(folder, fname)
        if 'train' in fname or 'test' in fname:
            print(file_path)
            df = pd.read_excel(file_path)
            df['proof'] = df['f7'].apply(proof_extract)
            df.to_excel(file_path.replace('.xlsx', '_proof.xlsx'), index=False)




def sample():
    pass
if __name__ == '__main__':
    folder = './data_excel/case_list_original_hetong_all_20210110'
    des_folder = './data_excel/case_list_original_hetong_all_20210110_sample2'
    des_path = os.path.join(des_folder, 'case_list_original_hetong_all_20210110_sample.csv')
    sample_list = {
        '房屋租赁合同纠纷': [500, 50],
        '租赁合同纠纷': [500, 50]
    }
    os.makedirs(des_folder, exist_ok=True)
    sample_from_excel(sample_list, folder, des_path)
    print('完成sample')
    split_train_test(folder, des_path, des_folder)
    print('完成分割训练集')
    extract_paragraph(des_folder)



