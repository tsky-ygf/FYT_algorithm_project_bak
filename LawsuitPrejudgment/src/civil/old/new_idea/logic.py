real_estate_division ='real_estate_division'


house = {
    1:{'id':1,'name':'房产','father':None,'children':[2,31],'status':False},
    2:{'id':2,'name':'婚前','father':1,'children':[3,12,21],'status':False},
    3: {'id':3,'name':'自己还清贷款或者全款购买','father':2,'children':[4,6,8],'status':False},
    4: {'id':4,'name':'登记在自己名下','father':3,'children':[5],'status':False},
    5: {'id':5,'name':'出资方个人财产','father':4,'children':None,'status':False},
    #------------------------------------------------------------------------
    6: {'id':6,'name':'登记在双方名下','father':3,'children':[7],'status':False},
    7: {'id':7,'name':'共同财产','father':6,'children':None,'status':False},
    #—————————————————————————————————————————————————————————————————————————
    8: {'id':8,'name':'登记在对方名下','father':3,'children':[9,10],'status':False},
    9: {'id':9,'name':'登记方个人财产','father':8,'children':None,'status':False},
    10:{'id':10,'name':'对方不具备购房资格，才已个人名义购房','father':8,'childrere':[11],'status':False},
    11:{'id':11,'name':'共同财产','father':10,'children':None,'status':False},

    12: {'id':12,'name':'婚后共同财产还贷','father':2,'children':[13,15,17],'status':False},
    13: {'id':13,'name':'登记在自己名下','father':12,'children':[14],'status':False},
    14: {'id':14,'name':'一般视为个人财产，夫妻共同还贷支付的款项以及房屋相对应的财产增值部分由双方平分而尚未偿还的贷款则为产权登记一方的个人债务','father':13,'children':None,'status':False},
    15: {'id':15,'name':'登记在双方名下','father':12,'children':[16],'status':False},
    16: {'id':16,'name':'共同财产','father':15,'children':None,'status':False},
    17: {'id':17,'name':'登记在对方名下','father':12,'children':[18,20],'status':False},
    18: {'id':18,'name':'对方不具备购房资格才用对方名义购房','father':17,'children':[18],'status':False},
    19: {'id':19,'name':'共同财产','father':18,'children':None,'status':False},
    20: {'id':20,'name':'登记方个人财产','father':17,'children':None,'status':False},


    21: {'id':21,'name':'一方父母(全额)出资','father':2,'children':[22,24],'status':False},
    22: {'id':22,'name':'登记在出资分子女名下','father':21,'children':[23],'status':False},
    23: {'id':23,'name':'登记方个人财产','father':22,'children':None,'status':False},
    24: {'id':24,'name':'一方父母支付了房屋首付款','father':21,'children':[25,27,29],'status':False},
    25: {'id':25,'name':'出资方子女名下','father':24,'children':[26],'status':False},
    26: {'id':26,'name':'一般判给登记方，对于婚内共同还贷部分（包括本金和利息）及其产生的增值，则由得房子的一方对另一方做出补偿','father':25,'children':None,'status':False},
    27: {'id':27,'name':'另一方子女名下','father':24,'children':[28],'status':False},
    28: {'id':28,'name':'共同财产','father':27,'children':None,'status':False},
    29: {'id':29,'name':'双方子女名下','father':24,'children':[30],'status':False},
    30: {'id':30,'name':'共同财产','father':29,'children':None,'status':False},

    31: {'id':31,'name':'婚后','father':1,'children':[32,41],'status':False},
    32: {'id':32,'name':'以个人财产买房','father':31,'children':[33,35,37,39],'status':False},
    33: {'id':33,'name':'登记在自己名下','father':32,'children':[34],'status':False},
    34: {'id':34,'name':'出资方个人财产','father':33,'children':None,'status':False},
    35: {'id':35,'name':'登记在对方名下','father':32,'children':[36],'status':False},
    36: {'id':36,'name':'共同财产','father':35,'children':None,'status':False},
    37: {'id':37,'name':'登记在子女名下','father':32,'children':[38],'status':False},
    38: {'id':38,'name':'子女的财产','father':37,'children':None,'status':False},
    39: {'id':39,'name':'登记在双方名下','father':32,'children':[40],'status':False},
    40: {'id':40,'name':'双方共同财产','father':39,'children':None,'status':False},

    41: {'id':40,'name':'以共同财产买房','father':31,'children':[42,44,46],'status':False},
    42: {'id':42,'name':'登记在一方名下','father':41,'children':[43],'status':False},
    43: {'id':43,'name':'共同财产','father':42,'children':None,'status':False},
    44: {'id':44,'name':'登记在双方名下','father':41,'children':[45],'status':False},
    45: {'id':45,'name':'共同财产','father':44,'children':None,'status':False},
    46: {'id':46,'name':'登记在子女名下','father':41,'children':[47],'status':False},
    47: {'id':47,'name':'子女财产','father':46,'children':None,'status':False}
}

logic_tree = {real_estate_division:house}