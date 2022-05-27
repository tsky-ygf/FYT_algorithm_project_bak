### 获取案由列表 
请求地址: `http://172.19.82.199:6021/getAnyou/`  
请求方式: `POST`    
请求参数: `{}`  
返回参数: `{
    "status": 0,
    "error_msg": "",    
    "AnyouList": [      
        "劳动社保_享受失业保险",  
        "劳动社保_享受医疗保险",  1
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

### 统计当天工作量  
请求地址: `http://172.19.82.199:6021/getWorkCount/`  
请求方式: `POST`    
请求参数: `{  
            "name":"汪丽浩",
         }`        
返回参数: `{
    "work_count": , 
    "error_msg": "",
    "status": 0, }
}`
异常参数: `{
    "error_msg": "this person have no work",
    "status": 1,
}`

### 二次审核  
请求地址: `http://172.19.82.199:6021/getSecondCheck/`  
请求方式: `POST`    
请求参数: `{  
            "anyou":"借贷纠纷_民间借贷",
         }`        
返回参数: `{
    "data_list": , 
    "error_msg": "",
    "status": 0, }
}`
异常参数: `{
    "error_msg": "no check",
    "status": 1,
}`

### 二次审核  确认
请求地址: `http://172.19.82.199:6021/getSecondCheckTrue/`  
请求方式: `POST`    
请求参数: `{  
            "suqiu":"",
            "jiufen_type":"",
            "source":"",
            "content":"",
            "mention":"",
            "situation":"",
            "startposition":"",
            "endpostion":"",
            "pos_neg":"",
            "labelingdate":"",
            "labelingperson":"",
            "checkperson":"",  
            "checkResult":"",      不填:二次审核正确，填写(可以填任何内容):审核有问题
            "id":"",
         }`        
返回参数: `{
    "result": update success, 
    "error_msg": "",
    "status": 0, }
}`
异常参数: `{
    "error_msg": "no check person",
    "status": 1,
}`


### 展示原文（目前没确定查询那张表）  
请求地址: `http://172.19.82.199:6021/getSourceContent/`  
请求方式: `POST`    
请求参数: `{  
            "key":"8d5e10b216d7402db6b4acbe00a676be",
         }`        
返回参数: `{
    "content": , 
    "error_msg": "",
    "status": 0, }
}`
异常参数: `{
    "error_msg": "have no data",
    "status": 1,
}`

### 添加新用户
请求地址: `http://172.19.82.199:6021/registerUser/`  
请求方式: `POST`    
请求参数: `{  
            "username":"",
            "password":"",
         }`        
返回参数: `{
    "content": "register success", 
    "error_msg": "",
    "status": 0, }
}`
异常参数1: `{
    "error_msg": "this user name is in use,repeat of user name",
    "status": 1,
}`
异常参数2: `{
    "error_msg": "no username or password",
    "status": 1,
}`
异常参数3: `{
    "error_msg": "data is not dictionary",
    "status": 1,
}`


### 登录校验
请求地址: `http://172.19.82.199:6021/loginCheck/`  
请求方式: `POST`    
请求参数: `{  
            "username":"",
            "password":"",
         }`        
返回参数: `{
    "content": "login success", 
    "error_msg": "",
    "status": 0, }
}`
异常参数1: `{
    "error_msg": "no username or password",
    "status": 1,
}`
异常参数2: `{
    "error_msg": "data is not dictionary",
    "status": 1,
}`
异常参数3: `{
    "error_msg": "Password error",
    "status": 1,
}`
异常参数4: `{
    "error_msg": "There is no such user",
    "status": 1,
}`
