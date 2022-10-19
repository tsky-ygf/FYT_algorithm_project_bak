# 算法API文档

在线文档地址：http://101.69.229.138:8130/

## 1. 合同智审
### 1.1 获取能审核的合同类型
`GET http://101.69.229.138:8131/get_contract_type`
### 1.2 获取甲乙方代表
`GET http://101.69.229.138:8131/get_user_standpoint`
### 1.3 获取模型审核结果
`POST http://101.69.229.138:8131/get_contract_review_result`
### 1.4 从文件链接获取文件文本内容
`POST http://101.69.229.138:8131/get_text_from_file_link_path`

## 2. 检索
## 3. 诉讼预判

### 3.1 获取民事预判支持的纠纷类型
`GET http://101.69.229.138:8133/get_civil_problem_summary`

输入参数:

无

输出参数:

@success: boolean, 是否成功调用

@error_msg: string, 失败时的错误信息

@value: array, 数据列表
    
* @groupName: string, 组名

* @groupList: List, 数据列表

  * @id: int, 纠纷id

  * @problem: string, 纠纷名称

### 3.2 获取民事纠纷对应的用户描述模板
`GET http://101.69.229.138:8133/get_template_by_problem_id`

输入参数:

@problem_id: int, 纠纷id

输出参数:

@success: boolean, 是否成功调用

@error_msg: string, 失败时的错误信息

@value: Dict, 数据字典
    
* @template: string, 用户描述模板

### 3.3 获取民事纠纷对应的诉求列表
`POST http://101.69.229.138:8133/get_claim_list_by_problem_id`

输入参数:

@problem_id: int, 纠纷id

@fact: string, 事实描述

输出参数:

@success: boolean, 是否成功调用

@error_msg: string, 失败时的错误信息

@value: List, 数据列表
    
  * @id: int, 诉求id

  * @claim: string, 诉求名称

  * @is_recommended: boolean, 是否推荐该诉求，示例：true。如果为true，则前端把该诉求设置为选中。

### 3.4 诉讼预判的主流程，包括民事、刑事和行政预判
`POST http://101.69.229.138:8133/lawsuit_prejudgment`

输入参数:

@dialogue_history: Dict, 对话历史
* @user_input: string, 事实描述
* @question_answers: List, 问答记录
  * @question: string, 问题
  * @candidate_answers: List, 候选项
  * @question_type: string, 问题类型，如“single”代表单选，“multiple”代表多选
  * @user_answer: List, 用户答案
  * @other: Dict, 其他辅助信息，不用做处理。

@dialogue_state: Dict, 对话状态
* @domain: string, 诉讼预判的类型，如'civil'代表民事, 'criminal'代表刑事, 'administrative'代表行政
* @problem: string, 纠纷类型
* @claim_list: List, 诉求列表
* @other: Dict, 其他辅助信息，不用做处理。

输出参数:

@success: boolean, 是否成功调用

@dialogue_history: Dict, 对话历史

@dialogue_state: Dict, 对话状态

@next_action: Dict, 下一次行动的信息
    
  * @action_type: string, 下一次行动的类型, 如'ask'代表提问, 'report'代表产生报告

  * @content: Dict, 下一次行动的内容
    
    如果是ask,内容为
      * @question: string, 问题
      * @candidate_answers: List, 候选项
      * @question_type: string, 问题类型，如“single”代表单选，“multiple”代表多选
      * @other: Dict, 其他辅助信息，不用做处理。

    如果是report,内容为
      * @report: List, 预判报告内容
      * @similar_case: List, 相似案例
      * @applicable_law: List, 相关法条
    
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