### 获取案由列表 
请求地址: `http://172.19.82.199:6021/getAnyou/`  
请求方式: `POST`    
请求参数: `{}`  
返回参数: `{
    "status": 0,
    "error_msg": "",    
    "AnyouList": [      
        "劳动社保_享受失业保险",  
        "劳动社保_享受医疗保险",  
        ...
    ]
}`

### 获取情形和特征列表  
请求地址: `http://172.19.82.199:6021/getCaseFeature/`  
请求方式: `POST`    
请求参数: `{"Anyou":"劳动社保_享受失业保险"}`    
返回参数: `{
    "status": 0,
    "error_msg": "",    
    "AnyouList": {      
        "劳动者自愿失业":  
            ['失业人员自愿失业']  
        ...
    }
}`


### 获取基础数据  
请求地址: `http://172.19.82.199:6021/getBaseData/`  
请求方式: `POST`    
请求参数: `{"Anyou":"借贷纠纷_民间借贷"}`    
返回参数: `{
    "status": 0,
    "error_msg": "",    
    "base_data": {      
        "case_id": "",
        "本院认为": "",
        "原告诉称": "",
        "本院查明": "",
        ...
    }
}`

### 获取预标注数据  
请求地址: `http://172.19.82.199:6021/getBaseAnnotation/`  
请求方式: `POST`    
请求参数: `{
            "Anyou":"借贷纠纷_民间借贷",
            "sentence":"",
            "contentHtml":""
         }`        
返回参数: `{
    "status": 0,
    "error_msg": "",
    "contentHtml": "",
    "base_data": {      
        [{
            'feature':'借款人逾期未偿还借款',
            'mention':'何三宇、冯群华迟迟不还款',
            'start_pos':75,
            'end_pos':87,
            'pos_or_neg':1',}
            ...
        ]       
        ...
    }
}`


### 插入标注结果  
请求地址: `http://172.19.82.199:6021/insertAnnotationData/`  
请求方式: `POST`    
请求参数: `{  
            "anyou_name":"借贷纠纷_民间借贷",
            "source":"原告诉称",
            "insert_data":[
                    {   
                        "id":"",
                        "content":"",
                        "situation":"存在借款合同",
                        "factor":"存在借款合同",
                        "start_pos":8,
                        "end_pos":20,
                        "pos_or_neg":1,
                },
            ...
                ]
         }`        
返回参数: `{
    "status": 0,
    "error_msg": "",
    "insert_result": "success", }
}`