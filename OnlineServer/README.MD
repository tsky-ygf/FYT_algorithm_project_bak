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
### 2.1 获取案例检索的过滤条件:
`GET http://101.69.229.138:8132/get_filter_conditions_of_case`
输入参数:

无
输出参数:

@result: dict, 案例检索的过滤条件

### 2.2 获取案例检索的结果:
`GET http://101.69.229.138:8132/search_cases`
输入参数:
@query: str, 用户输入的查询内容

@filter_conditions: dict, 用户输入的条件

    @type_of_case: list, 用户输入的案件类型

    @court_level: list, 用户输入的法院层级

    @type_of_document: list, 用户输入的文书类型

    @region: list, 用户输入的地域

@page_number: int, 第几页

@page_size: int, 页大小

输出参数:

@result: list, 搜索的返回结果

@total_amount: int 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度

### 2.3 获取法律检索的过滤条件:
`GET http://101.69.229.138:8132/get_filter_conditions_of_law`
输入参数:

无
输出参数:

@result: dict, 法律检索的过滤条件

### 2.4 获取法律检索的结果:
`GET http://101.69.229.138:8132/search_laws`
输入参数:
@query: str, 用户输入的查询内容

@filter_conditions: dict, 用户输入的条件

    @timeliness: str,  用户输入的时效性

    @types_of_law：str, 用户输入的法律种类

    @scope_of_use：str, 用户输入的使用范围

@page_number: int, 第几页

@page_size: int, 页大小

输出参数:

@result: list, 搜索的返回结果

@total_amount: int 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度

### 2.5 获取相似类案的结果:
`GET http://101.69.229.138:8132/top_k_similar_narrative`
输入参数:

@fact: str, 用户输入的事实描述

@problem: str, 用户输入的纠纷类型

@claim_list: list, 诉求类型

输出参数:

@dids: list, 裁判文书的uq_id

@sims: list, 相似类案的相似率

@reasonNames: list, 纠纷类型

@tags: list, 关键词

### 2.6 获取文本人名脱敏的结果:
`GET http://101.69.229.138:8132/desensitization`
输入参数:

@text: str, 待脱敏的文本

输出参数:

@result: str, 人名脱敏后的文本

## 3. 诉讼预判
## 4. 智能咨询
### 4.1 获取智能咨询的结果:
`POST http://101.69.229.138:8134/get_query_answer_with_source`
输入参数:

@question: str , 用户输入的问题

@query_type: str, 选择一级主题分类，可选项:(旧版/专题)

@query_sub_type: str, 选择二级主题分类，可选项:(通用/法院/公安/环保/交通/金融/科技/市场监管/税务/司法/文旅)

输出参数:

@answer: str, 智能咨询的答案

@similarity_question: Optional[List]: 返回的相似问题

## 5.纠错模块
### 5.1 获取输入文本的纠错结果及详细信息
`POST http://101.69.229.138:8150/get_corrected_contract_result`

输入参数：

@text: str, 输入文本

输出参数：

@result:dict, 文本的纠错结果及详细信息
