from logic import logic_tree
print(logic_tree['real_estate_division'])

logic = logic_tree['real_estate_division']
features = []
for i in range(1,len(logic)+1):
    features.append(logic[i]['name'])
features = list(set(features))

logic[1]['status'] = True
path = []
current_node = logic[1]


while current_node['children'] != None:
    current_children_status = []
    current_children_names = []
    for child in current_node['children']:
        current_children_status.append(logic[child]['status'])
        current_children_names.append(logic[child]['name'])
    if True in current_children_status:
        index = current_children_status.index(True)
        id = current_node['children'][index]
        current_node = logic[id]
    else:#提问环节
        QA_node = [] #存放当前node的所有孩子节点
        leaf_node = []
        print_info = []
        if logic[current_node['children'][0]]['children'] == None:
            print('============打印推理结果=============')
            print("推理结果："+logic[current_node['children'][0]]['name'])
            break

        else:
            for child in current_node['children']:#遍历得到当前node的所有子节点
                QA_node.append(logic[child])
            for i,node in enumerate(QA_node):#为了可视化处理当前node的虽有子节点的信息
                print_info.append(str(i)+':'+node['name'])
            print(' '.join(print_info))#打印所有子节点的信息
            index = int(input('请根据您实际情况选择对应的编号:'))#用户输入子节点的编号
            current_node = QA_node[index]
