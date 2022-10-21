# 爬虫API文档1.0

[//]: # (在线文档地址：http://101.69.229.138:8130/)

## 1. 使用requests方式请求网站的爬虫

### 1.1 获取能审核的合同类型

`GET http://101.69.229.138:8131/get_contract_type`

请求参数:

无

响应参数：

| Param  | Type       | Description  |
|--------|------------|--------------|
| result | List[Dict] | 支持的合同类型   |

result包含若干个如下的dict：
   * type_id : str 合同类型的id  
   * contract_type : str 合同类型



## 2. 使用Scrapy框架请求网站的爬虫

### 2.1 获取案例检索的过滤条件:

`GET http://101.69.229.138:8132/get_filter_conditions_of_case`

请求参数:

无

响应参数:

| Param            | Type | Description |
|------------------|------|-------------|
| type_of_case     | Dict | 案例检索的过滤条件   |
| court_level      | Dict | 案例检索的过滤条件   |
| type_of_document | Dict | 案例检索的过滤条件   |
| region           | Dict | 案例检索的过滤条件   |

type_of_case的内容如下:
  * name: str, 案件类型

  * is_multiple_choice: boolean, 是否支持多选
  
  * value: List[str], 案件类型的选项，选项（全部/刑事/民事/行政/执行）

court_level的内容如下:

  * name: str, 法院层级

  * is_multiple_choice: boolean, 是否支持多选
  
  * value: List[str], 法院层级的选项，选项（全部/最高/高级/中级/基层）

type_of_document的内容如下:

  * name: str, 文书类型

  * is_multiple_choice: boolean, 是否支持多选
  
  * value: List[str], 文书类型的选项，选项（全部/判决/裁定/调解）

region的内容如下:

  * name: str, 地域

  * is_multiple_choice: boolean, 是否支持多选
  
  * value: List[str], 地域的选项，选项（全国/安徽省/北京市/重庆市/福建省/甘肃省/广东省/
  广西壮族自治区/贵州省/海南省/河北省/河南省/黑龙江省/湖北省/湖南省/吉林省/江苏省/江西省/
  辽宁省/内蒙古自治区/宁夏回族自治区/青海省/山东省/山西省/陕西省/上海市/四川省/天津市/
  西藏自治区/新疆维吾尔自治区/云南省/浙江省）
