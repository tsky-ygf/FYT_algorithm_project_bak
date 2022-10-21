#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 08:57
# @Author  : Adolf
# @Site    :
# @File    : server.py
# @Software: PyCharm
import json
import typing

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union
from ProfessionalSearch.src.similar_case_retrival.similar_case.narrative_similarity_predict import (
    predict_fn as predict_fn_similar_cases,
)

from ProfessionalSearch.service_use.relevant_laws.relevant_laws_api import (
    get_filter_conditions,
    _get_law_result,
)
from ProfessionalSearch.service_use.similar_case_retrival.similar_case_retrieval_api import (
    get_filter_conditions_of_case,
    _get_case_result,
)
from ProfessionalSearch.src.similar_case_retrival.similar_case.util import (
    desensitization,
)
from Utils.http_response import response_successful_result, response_failed_result

app = FastAPI()


class Filter_conditions_law(BaseModel):
    timeliness: list = Field(description="时效性")
    types_of_law: list = Field(description="法律种类")
    scope_of_use: list = Field(description="地域")


class Search_laws_input(BaseModel):
    query: str = Field(description="用户输入的查询内容")
    filter_conditions: Union[Filter_conditions_law, None] = None
    page_number: int = Field(description="页")
    page_size: int = Field(description="页大小")

    class Config:
        schema_extra = {
            "example": {
                "query": "侵权",
                "filter_conditions": {
                    "types_of_law": ["地方性法规"],
                    "timeliness": ["全部"],
                    "scope_of_use": ["广东省"],
                },
                "page_number": 1,
                "page_size": 10,
            }
        }


class Law_id(BaseModel):
    law_id: str


class Filter_conditions_case(BaseModel):
    type_of_case: list = Field(description="案例类型")
    court_level: list = Field(description="法院层级")
    type_of_document: list = Field(description="文书类型")
    region: list = Field(description="地域")


class Search_cases_input(BaseModel):
    query: str = Field(description="用户输入的检索内容")
    filter_conditions: Union[Filter_conditions_case, None] = None
    page_number: int = Field(description="页")
    page_size: int = Field(description="页大小")

    class Config:
        schema_extra = {
            "example": {
                "page_number": 1,
                "page_size": 10,
                "query": "买卖",
                "filter_conditions": {
                    "type_of_case": ["民事"],
                    "court_level": ["中级"],
                    "type_of_document": ["裁定"],
                    "region": ["安徽省"],
                },
            }
        }


class Similar_case_input(BaseModel):
    fact: str = Field(description="用户输入的查询内容")
    problem: str = Field(description="纠纷类型")
    claim_list: list = Field(description="诉求类型")

    class Config:
        schema_extra = {
            "example": {"problem": "民间借贷纠纷", "claim_list": [], "fact": "借钱不还"}
        }


class Case_id(BaseModel):
    case_id: str


class conditions_law_result(BaseModel):
    types_of_law: typing.Dict = Field(description="法律种类")
    timeliness: typing.Dict = Field(description="时效性")
    scope_of_use: typing.Dict = Field(description="地域")


class laws(BaseModel):
    law_id: str = Field(description="数据库表明+_SEP_+法条的md5Clause")
    law_name: str = Field(description="标题")
    law_type: str = Field(description="国家法律法规网页标题")
    timeliness: str = Field(description="时效性")
    using_range: str = Field(description="省")
    law_chapter: str = Field(description="法条章节")
    law_item: str = Field(description="法条条目")
    law_content: str = Field(description="法条内容")


class search_laws_result(BaseModel):
    laws: typing.List[laws]
    total_amount: int = Field(description="返回总量, 大于200条，默认200")


class conditions_case_result(BaseModel):
    type_of_case: dict = Field(description="案例类型")
    court_level: dict = Field(description="法院层级")
    type_of_document: dict = Field(description="文书类型")
    region: dict = Field(description="地域")


class cases(BaseModel):
    doc_id: str = Field(description=" +数据库表名+ _SEP_ + uq_id")
    court: str = Field(description="法院名字")
    case_number: str = Field(description="文书号")
    jfType: str = Field(description="纠纷类型")
    content: str = Field(description="裁判文书内容")


class search_cases_result(BaseModel):
    cases: typing.List[cases]
    total_amount: int = Field(description="案例返回的总量，大于200， 默认200")


class similar_narrative_result(BaseModel):
    dids: list[str] = Field(description="id集合")
    sims: list[float] = Field(description="相似率")
    reasonNames: list[str] = Field(description="纠纷类型")
    tags: list[str] = Field(description="关键词")


class desensitization_result(BaseModel):
    res: str = Field(description="脱敏人名后的文本")


class desensitization_input(BaseModel):
    text: str = Field(description="待人名脱敏的文本")

    class Config:
        schema_extra = {
            "example": {

                "text": "《八佰》（英語：The Eight Hundred）是一部于2020年上映的以中国历史上的战争为题材的电影，由管虎执导，黄志忠、黄骏豪、张俊一、张一山....."

            }
        }


@app.get("/get_filter_conditions_of_law", response_model=conditions_law_result)
async def _get_filter_conditions() -> dict:
    """
    返回法条检索的输入条件

    请求参数:

    无

    响应参数:

    | Param        | Type | Description |
    |--------------|------|-------------|
    | types_of_law | Dict | 法条种类        |
    | timeliness   | Dict | 时效性         |
    | scope_of_use | Dict | 使用范围        |

    types_of_law的内容如下:

      * name: str, 法条种类

      * is_multiple_choice: boolean, 是否支持多选

      * value: List[str], 法条种类的选项，选项（全部/宪法/法律/行政法规/监察法规/司法解释/地方性法规 ）

    timeliness的内容如下:

      * name: str, 时效性

      * is_multiple_choice: boolean, 是否支持多选

      * value: List[str], 时效性的选项，选项（全部/有效/已修改/尚未生效/已废止）

    scope_of_use的内容如下:

      * name: str, 使用范围

      * is_multiple_choice: boolean, 是否支持多选

      * value: List, 使用范围的选项，选项（全国/安徽省/北京市/重庆市/福建省/甘肃省/广东省/
      广西壮族自治区/贵州省/海南省/河北省/河南省/黑龙江省/湖北省/湖南省/吉林省/江苏省/江西省/
      辽宁省/内蒙古自治区/宁夏回族自治区/青海省/山东省/山西省/陕西省/上海市/四川省/天津市/
      西藏自治区/新疆维吾尔自治区/云南省/浙江省）
    """
    return get_filter_conditions()


@app.post("/search_laws", response_model=search_laws_result)
async def search_laws(search_query: Search_laws_input) -> dict:
    """
    获取法律检索的结果

    请求参数:

    | Param             | Type | Description          |
    |-------------------|------|----------------------|
    | query             | str  | 用户输入的查询内容            |
    | filter_conditions | Dict | 用户输入的条件，小于就返回搜索结果的长度 |
    | page_number       | int  | 第几页                  |
    | page_size         | int  | 页大小                  |

    filter_conditions的内容如下:

    * timeliness: List[str], 时效性，可选项（全部/有效/已修改/尚未生效/已废止）

    * types_of_law: List[str], 法条种类，可选项（全部/宪法/法律/行政法规/监察法规/司法解释/地方性法规 ）

    * scope_of_use: List[str], 地域，可选项（全国/安徽省/北京市/重庆市/福建省/甘肃省/广东省/
      广西壮族自治区/贵州省/海南省/河北省/河南省/黑龙江省/湖北省/湖南省/吉林省/江苏省/江西省/
      辽宁省/内蒙古自治区/宁夏回族自治区/青海省/山东省/山西省/陕西省/上海市/四川省/天津市/
      西藏自治区/新疆维吾尔自治区/云南省/浙江省）

    响应参数:

    | Param        | Type       | Description                      |
    |--------------|------------|----------------------------------|
    | laws         | List[dict] | 返回的法律list集合                      |
    | total_amount | int        | 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度 |

    laws的内容如下:

    *   law_id:        str,    数据库表名 + 分隔符 + 法条的md5Clause
    *   law_name:      str,    标题
    *   law_type:      str,    国家法律法规网页标题
    *   timeliness:    str,    时效性
    *   using_range:   str,    省份
    *   law_chapter:   str,    法条章节
    *   law_item:      str,    法条条目
    *   law_content:   str,    法条内容

    """
    try:
        result = _get_law_result(
            search_query.query,
            search_query.filter_conditions,
            search_query.page_number,
            search_query.page_size,
        )
        return result
    except Exception as e:
        return response_failed_result("error:" + repr(e))


# @app.get("/get_law_by_law_id")
# def get_law_by_law_id(law_id: Law_id):
#     # TODO 待更新
#     return {"result": ""}


@app.get("/get_filter_conditions_of_case", response_model=conditions_case_result)
async def _get_filter_conditions() -> dict:
    """
    返回案例检索的过滤条件

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
    """
    return get_filter_conditions_of_case()


@app.post("/search_cases", response_model=search_cases_result)
async def search_cases(search_query: Search_cases_input) -> dict:
    """
    获取法律检索的结果

    请求参数:

    | Param             | Type | Description |
    |-------------------|------|-------------|
    | query             | str  | 用户输入的查询内容   |
    | filter_conditions | Dict | 用户输入的条件     |
    | page_number       | int  | 第几页         |
    | page_size         | int  | 页大小         |

    filter_conditions的内容如下:

    * type_of_case: List[str], 案件类型，可选项（全部/刑事/民事/行政/执行）

    * court_level: List[str], 法院层级，可选项（全部/最高/高级/中级/基层）

    * type_of_document: List[str], 文书类型，可选项（全部/判决/裁定/调解）

    * region: List[str], 地域，可选项（全国/安徽省/北京市/重庆市/福建省/甘肃省/广东省/
      广西壮族自治区/贵州省/海南省/河北省/河南省/黑龙江省/湖北省/湖南省/吉林省/江苏省/江西省/
      辽宁省/内蒙古自治区/宁夏回族自治区/青海省/山东省/山西省/陕西省/上海市/四川省/天津市/
      西藏自治区/新疆维吾尔自治区/云南省/浙江省）

    响应参数:

    | Param        | Type       | Description                      |
    |--------------|------------|----------------------------------|
    | cases        | List[dict] | 返回案例的list集合                      |
    | total_amount | int        | 若搜索的结果有200条以上则返回200，小于就返回搜索结果的长度 |

    cases的内容如下:

    * doc_id:        str,  数据库表名 + 分隔符 + 裁判文书的uq_id
    * court:         str,  法院名称
    * case_number:   str,  文书号
    * jfType:        str,  纠纷类型
    * content:       str,  裁判文书内容
    """
    try:
        if search_query is not None:
            query = search_query.query
            filter_conditions = search_query.filter_conditions
            page_number = search_query.page_number
            page_size = search_query.page_size
            if query is not None and filter_conditions is not None:
                result = _get_case_result(
                    query, filter_conditions, page_number, page_size
                )
                # 返回数量，若200以上，则返回200，若小于200，则返回实际number
                return result
            else:
                return response_successful_result([], {"total_amount": len([])})
        else:
            return {"error_msg": "no data", "status": 1}
    except Exception as e:
        return response_failed_result("error:" + repr(e))


# @app.get("/get_law_document")
# def get_law_document(case_id: Case_id):
#     # TODO 待更新
#     return {"result": ""}


@app.post("/top_k_similar_narrative", response_model=similar_narrative_result)
async def get_similar_case(search_query: Similar_case_input) -> dict:
    """
    返回相似类案的结果

    请求参数:

    | Param      | Type | Description                                                                                                                                                                                                                                                                                                                                                                                                                                               |
    |------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | fact       | str  | 用户输入的事实描述                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
    | problem    | str  | 用户输入的纠纷类型，可选项:(婚姻家庭/继承纠纷/子女抚养/老人赡养/返还彩礼/财产分割/同居问题/工伤赔偿/社保纠纷/劳动纠纷/劳务纠纷/劳务人员受害/劳务人员致害/用人单位责任/民间借贷/金融借贷/交通事故纠纷/人身伤害赔偿/财产损失赔偿/保险赔付/医药费赔偿/误工费赔偿/违章扣分/合伙纠纷/股权转让/买卖合同/租赁合同/运输合同/赠与合同/承揽合同/委托合同/保证合同/抵押合同/技术服务合同/技术开发合同/技术转让合同/技术咨询合同/中介合同/融资租赁合同/不当得利/善意取得/房屋买卖/物业服务/仓储合同/相邻关系/业主权/网络侵权/隐私权和个人信息保护/名誉权与荣誉权/肖像权/姓名权/动物致人损害/触电人身损害/道路施工损害/道路通行损害/建筑倒塌损害/物件脱落损害/高空抛物/医疗损害/公共场所损害/监护人责任/教育机构责任/人身损害/消费者权利保护/产品责任/建设用地使用权/建设工程合同/土地承包/宅基地使用权/著作权纠纷/专利权纠纷/商标权纠纷/不正当竞争) |
    | claim_list | List | 诉求类型                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

    响应参数:

    | Param       | Type        | Description |
    |-------------|-------------|-------------|
    | dids        | List[str]   | 裁判文书的uq_id  |
    | sims        | List[float] | 相似类案的相似率    |
    | reasonNames | List[str]   | 纠纷类型        |
    | tags        | List[str]   | 关键词         |
    """
    try:
        if search_query is not None:
            fact = search_query.fact
            problem = search_query.problem
            claim_list = search_query.claim_list
            (
                doc_id_list,
                sim_list,
                # win_los_list,
                reason_name_list,
                appeal_name_list,
                tags_list,
                keywords,
                pubDate_list,
            ) = predict_fn_similar_cases(fact, problem, claim_list)

            return {
                "dids": doc_id_list,
                "sims": sim_list,
                # "winLos": win_los_list,
                "reasonNames": reason_name_list,
                # "appealNames": appeal_name_list,
                "tags": tags_list,
                # "pubDates": pubDate_list,
                # "keywords": keywords,
                # "error_msg": "",
                # "status": 0,
            }

        else:
            return {"error_msg": "data is None", "status": 1}
    except Exception as e:
        return {"error_msg:": repr(e), "status": 1}


@app.post("/desensitization", response_model=desensitization_result)
async def get_text_desen(dese_in: desensitization_input) -> dict:
    """
    返回人名脱敏的结果

    请求参数:

    | Param | Type | Description |
    |-------|------|-------------|
    | text  | str  | 待脱敏的文本      |

    响应参数:

    | Param  | Type | Description |
    |--------|------|-------------|
    | result | str  | 人名脱敏后的文本    |
    """
    return {"res": desensitization(dese_in.text)}


if __name__ == "__main__":
    # 日志设置
    uvicorn.run(
        "OnlineServer.ProfessionalSearch.server:app",
        host="0.0.0.0",
        port=8132,
        reload=False,
        workers=1,
    )
