# 算法API文档

在线文档地址：http://101.69.229.138:8130/

## 1. 合同智审

### 1.1 获取能审核的合同类型

`GET http://101.69.229.138:8131/get_contract_type`

输入参数:

无

输出参数：

@result：支持的合同类型

type：list 里包含若干个dict

### 1.2 获取甲乙方代表

`GET http://101.69.229.138:8131/get_user_standpoint`

输入参数：

无

输出参数：

@result：获取双方代表

type：list 里包含两个dict

### 1.3 获取合同审核结果

`POST http://101.69.229.138:8131/get_contract_review_result`

输入参数：

@contract_type_id：合同类型

@usr： 甲乙方代表

@contract_content： 合同内容

输出参数：

@result：合同审核结果

type：list

### 1.4 从文件链接获取文件文本内容

`POST http://101.69.229.138:8131/get_text_from_file_link_path`

输入参数：

@file_path: 文件链接

输出参数：

@result:文件文本内容

type：str

## 2. 检索

## 3. 诉讼预判

## 4. 智能咨询

### 4.1 获取智能咨询的结果:

`POST http://101.69.229.138:8134/get_query_answer_with_source`

请求参数:

| Param           | Type | Description                                       |
|-----------------|------|---------------------------------------------------|
| question        | str  | 用户输入的问题                                           |
| query_type      | str  | 选择一级主题分类，可选项:(旧版/专题)                              |
| query_sub_type  | str  | 选择二级主题分类，可选项:(通用/法院/公安/环保/交通/金融/科技/市场监管/税务/司法/文旅) |

响应参数:

| Param               | Type           | Description |
|---------------------|----------------|-------------|
| answer              | str            | 智能咨询的答案     |
| similarity_question | Optional[List] | 返回的相似问题     |