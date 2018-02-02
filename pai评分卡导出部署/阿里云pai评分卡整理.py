# -*- coding: utf-8 -*-
from odps import ODPS
from odps.models import Schema, Column, Partition
from odps.inter import setup, enter, teardown
from odps.df import DataFrame
import pandas as pd
import numpy as np
import re 

# ODPS账号
access_id = ''
access_key = ''
project = 'Modelgroup'
o = ODPS(access_id, access_key, project)

# 入参特征
key = ['onlinebank_credit_bill_cnt_lessminpay_mths_9mth' ,
'onlinebank_credit_bill_maxratio_amt_cur_overdue_6mth' ,
'onlinebank_credit_bill_min_amt_cur_repay_6mth' ,
'onlinebank_credit_bill_max_amt_cur_overdue_9mth' ,
'onlinebank_billitem_cnt_overduepayment_months_9mth' ,
'onlinebank_credit_bill_cnt_lessbalance_bills_6mth' ,
'payamt_4' ,
'payamt_5' ,
'payamt_6' ,
'onlinebank_credit_bill_max_amt_cur_repay_6mth' ,
'onlinebank_credit_bill_min_amt_cur_charge_6mth' ,
'onlinebank_credit_bill_minratio_amt_cur_overdue_9mth' ,
'onlinebank_credit_installment_cnt_bill_service_6mth' ,
'onlinebank_credit_bill_max_amt_cur_minrepay_6mth' ,
'onlinebank_bill_item_sum_income_accrual_debitcard_9m' ,
'onlinebank_billitem_avg_amount_shangchao_9mth' ,
'onlinebank_credit_bill_onecard_min_sumratio_balance_credit_limit_9mth' ,
'onlinebank_credit_bill_min_amt_cur_minrepay_6mth' ]

# 数据处理
t = DataFrame(o.get_table('zeus_athena_onlinebank_scorecard_test')).to_pandas()
NONE_VIN = t['bin'].isnull()
data = t[['feaname','bin','woe','scaled_weight']].drop(t[t['bin'] == 'ELSE'].index).drop(t[NONE_VIN].index)

# 格式处理
for i in key:
    cutpoint = []
    cut = []
    
    df = data[data['feaname'] == i][['bin','woe','scaled_weight']]
    df = df.replace('NULL',np.nan)   
    df.rename(columns={'bin': i + '_cut','woe': 'WOE','scaled_weight': 'point'}, inplace=True) 
       
    a = list(df[i + '_cut'])
    a.pop(-1)
    for j in a:
        reg = re.compile('[0-9]+[\.]?[0-9]*[e]?[\+]?[0-9]*')
        if a[0] == j:
            b = list(map(np.float, re.findall(reg, j)))
            b.insert(0, -np.inf)
            col2 = str(b).replace('[','(')
        elif a[-1] == j:
            b = list(map(np.float, re.findall(reg, j)))
            b.append(np.inf)
            col2 = str(b).replace('[','(')
        else:
            col2 = str(list(map(np.float, re.findall(reg, j)))).replace('[','(')
        coll = list(map(float, re.findall(reg, j)))
        cutpoint.append(coll[-1])
        cut.append(col2)
    cutpoint.pop(-1)
    cutpoint.insert(0, -np.inf)
    cutpoint.append(np.inf)   
    cut.append(np.nan)
    
    df['cut_point'] = str(cutpoint)  
    df[i + '_cut'] = cut
    df.to_excel('C:/Users/Administrator/Desktop/point/%s_point.xls' % i, header=True)
    

