import xmind
workbook = xmind.load('./xmind_files/房产归属0506.xmind')
data = workbook.getData()

#统计所有特征
all_feature = []
queue = []
#遍历所有的非叶子节点
current_node = data[0]
queue.append(current_node)
count = 1
#获取该棵树下所有的特征
while len(queue) !=0:
    current_node = queue.pop(0)
    all_feature.append(current_node['title'])
    if count == 1:
        current_node = current_node['topic']
        queue.append(current_node)
        current_layer = current_node['topics']
        count +=1
    else:
        current_layer = current_node['topics']
    if current_layer[0]['label'] == None:
        pass
    else:
        for node in current_layer:
            queue.append(node)
print(set(all_feature))

#第一层定位到房产
current_layer = data[0]['topic']['topics']

while True:
    label_flag = []
    for node in current_layer:
        label_flag.append(node['label'])
    if '1' in label_flag:
        index = label_flag.index('1')
        current_node = current_layer[index]
        current_layer = current_node['topics']
    else:
        titles = []
        for index,item in enumerate(current_layer):
            titles.append(str(index)+':'+item['title'])
        print('  '.join(titles))
        index = int(input('定输入对应编号:'))
        current_node = current_layer[index]
        current_layer = current_node['topics']
        if current_layer[0]['label'] == None:
            print("=============================评估理由=================================")
            print()
            print(current_layer[0]['title'])
            break