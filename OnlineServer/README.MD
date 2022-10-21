# 算法API文档1.0

在线文档地址：http://101.69.229.138:8130/

## 1. 合同智审

### 1.1 获取能审核的合同类型

`GET http://101.69.229.138:8131/get_contract_type`

请求参数:

无

响应参数：

| Param  | Type       | Description  |
|--------|------------|--------------|
| result | List[Dict] | 支持的合同类型      |

### 1.2 获取甲乙方代表

`GET http://101.69.229.138:8131/get_user_standpoint`

请求参数：

无

响应参数：

| Param  | Type       | Description  |
|--------|------------|--------------|
| result | List[Dict] | 合同审核结果       |

### 1.3 获取合同审核结果

`POST http://101.69.229.138:8131/get_contract_review_result`

请求参数：

| Param             | Type  | Description  |
|-------------------|-------|--------------|
| contract_type_id  | str   | 合同类型         |
| filter_conditions | str   | 甲乙方代表        |
| type_of_case      | str   | 合同内容         |
| court_level       | list  | 用户输入的法院层级    |

响应参数：

| Param  | Type  | Description  |
|--------|-------|--------------|
| result | list  | 合同审核结果       |

### 1.4 从文件链接获取文件文本内容

`POST http://101.69.229.138:8131/get_text_from_file_link_path`

请求参数：

| Param     | Type  | Description  |
|-----------|-------|--------------|
| file_path | str   | 文件链接         |

响应参数：

| Param  | Type  | Description  |
|--------|-------|--------------|
| result | str   | 文件文本内容       |

## 2. 检索

### 2.1 获取案例检索的过滤条件:

`GET http://101.69.229.138:8132/get_filter_conditions_of_case`

请求参数:

无

响应参数:

| Param  | Type  | Description  |
|--------|-------|--------------|
| result | dict  | 案例检索的过滤条件    |

### 2.2 获取案例检索的结果:

`GET http://101.69.229.138:8132/search_cases`

请求参数:

| Param             | Type  | Description  |
|-------------------|-------|--------------|
| query             | str   | 用户输入的查询内容    |
| filter_conditions | dict  | 用户输入的条件      |
| type_of_case      | list  | 用户输入的案件类型    |
| court_level       | list  | 用户输入的法院层级    |
| type_of_document  | list  | 用户输入的文书类型    |
| region            | list  | 用户输入的地域      |
| page_number       | int   | 第几页          |
| page_size         | int   | 页大小          |

响应参数:

| Param        | Type  | Description                      |
|--------------|-------|----------------------------------|
| result       | list  | 搜索的返回结果                          |
| total_amount | int   | 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度 |

### 2.3 获取法律检索的过滤条件:

`GET http://101.69.229.138:8132/get_filter_conditions_of_law`
请求参数:

无

响应参数:

| Param  | Type  | Description  |
|--------|-------|--------------|
| result | dict  | 法律检索的过滤条件    |

### 2.4 获取法律检索的结果:

`GET http://101.69.229.138:8132/search_laws`

请求参数:

| Param             | Type  | Description          |
|-------------------|-------|----------------------|
| query             | str   | 用户输入的查询内容            |
| filter_conditions | dict  | 用户输入的条件，小于就返回搜索结果的长度 |
| timeliness        | str   | 用户输入的时效性             |
| types_of_law      | str   | 用户输入的法律种类            |
| scope_of_use      | str   | 用户输入的使用范围            |
| page_number       | int   | 第几页                  |
| page_size         | int   | 页大小                  |

响应参数:

| Param        | Type  | Description                      |
|--------------|-------|----------------------------------|
| result       | list  | 搜索的返回结果                          |
| total_amount | int   | 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度 |

### 2.5 获取相似类案的结果:

`GET http://101.69.229.138:8132/top_k_similar_narrative`

请求参数:

| Param      | Type | Description |
|------------|------|-------------|
| fact       | str  | 用户输入的事实描述   |
| problem    | str  | 用户输入的纠纷类型   |
| claim_list | list | 诉求类型        |

响应参数:

| Param       | Type | Description     |
|-------------|------|-----------------|
| dids        | list | 裁判文书的uq_id      |
| sims        | list | 相似类案的相似率        |
| reasonNames | list | 纠纷类型            |
| tags        | list | 关键词             |

### 2.6 获取文本人名脱敏的结果:

`GET http://101.69.229.138:8132/desensitization`

请求参数:

| Param    | Type  | Description     |
|----------|-------|-----------------|
| text     | str   | 待脱敏的文本          |

响应参数:

| Param   | Type   | Description    |
|---------|--------|----------------|
| result  | str    | 人名脱敏后的文本       |

## 3. 诉讼预判

### 3.1 获取民事预判支持的纠纷类型
`GET http://101.69.229.138:8133/get_civil_problem_summary`

请求参数:

无

相应参数:

| Param     | Type    | Description  |
|-----------|---------|--------------|
| success   | boolean | 是否成功调用       |
| error_msg | string  | 失败时的错误信息     |
| value     | List    | 数据列表         |

value的内容如下:

* groupName: string, 组名

* groupList: List, 数据列表

  * id: int, 纠纷id

  * problem: string, 纠纷名称

### 3.2 获取民事纠纷对应的用户描述模板
`POST http://101.69.229.138:8133/get_template_by_problem_id`

请求参数:

| Param      | Type | Description |
|------------|------|-------------|
| problem_id | int  | 纠纷id        |

响应参数:

| Param     | Type    | Description |
|-----------|---------|-------------|
| success   | boolean | 是否成功调用      |
| error_msg | string  | 失败时的错误信息    |
| value     | Dict    | 数据字典        |

value的内容如下:

* template: string, 用户描述模板

### 3.3 获取民事纠纷对应的诉求列表
`POST http://101.69.229.138:8133/get_claim_list_by_problem_id`

请求参数:


| Param      | Type   | Description |
|------------|--------|-------------|
| problem_id | int    | 纠纷id        |
| fact       | string | 事实描述        |


响应参数:


| Param     | Type    | Description |
|-----------|---------|-------------|
| success   | boolean | 是否成功调用      |
| error_msg | string  | 失败时的错误信息    |
| value     | List    | 数据列表        |

value的内容如下:
    
  * id: int, 诉求id

  * claim: string, 诉求名称

  * is_recommended: boolean, 是否推荐该诉求，示例：true。如果为true，则前端把该诉求设置为选中。

### 3.4 诉讼预判的主流程，包括民事、刑事和行政预判
`POST http://101.69.229.138:8133/lawsuit_prejudgment`

请求参数:


| Param            | Type | Description |
|------------------|------|-------------|
| dialogue_history | Dict | 对话历史        |
| dialogue_state   | Dict | 对话状态        |

dialogue_history的内容如下:
* user_input: string, 事实描述
* question_answers: List, 问答记录
  * question: string, 问题
  * candidate_answers: List, 候选项
  * question_type: string, 问题类型，如“single”代表单选，“multiple”代表多选
  * user_answer: List, 用户答案
  * other: Dict, 其他辅助信息，不用做处理。

dialogue_state的内容如下:
* domain: string, 诉讼预判的类型，如'civil'代表民事, 'criminal'代表刑事, 'administrative'代表行政
* problem: string, 纠纷类型
* claim_list: List, 诉求列表
* other: Dict, 其他辅助信息，不用做处理。

响应参数:


| Param            | Type    | Description |
|------------------|---------|-------------|
| success          | boolean | 是否成功调用      |
| dialogue_history | Dict    | 对话历史        |
| dialogue_state   | Dict    | 对话状态        |
| next_action      | Dict    | 下一次行动的信息    |

next_action的内容如下:
    
  * action_type: string, 下一次行动的类型, 如'ask'代表提问, 'report'代表产生报告

  * content: Dict, 下一次行动的内容
    
    如果是ask,内容为
      * question: string, 问题
      * candidate_answers: List, 候选项
      * question_type: string, 问题类型，如“single”代表单选，“multiple”代表多选
      * other: Dict, 其他辅助信息，不用做处理。

    如果是report,内容为
      * report: List, 预判报告内容
      * similar_case: List, 相似案例
      * applicable_law: List, 相关法条
    
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

## 5.纠错模块

### 5.1 获取输入文本的纠错结果及详细信息

`POST http://101.69.229.138:8150/get_corrected_contract_result`

请求参数：

| Param    | Type  | Description   |
|----------|-------|---------------|
| text     | str   | 输入文本          |

响应参数：

| Param   | Type   | Description    |
|---------|--------|----------------|
| result  | dict   | 文本的纠错结果及详细信息   |
