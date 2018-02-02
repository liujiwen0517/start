from flask import Flask, request
import pandas as pd
import re
import numpy as np
import json
import logging

logger = logging.getLogger("point")

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
file_handler = logging.FileHandler("./point.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

app = Flask(__name__)

feature_final = [
'onlinebank_credit_bill_cnt_lessminpay_mths_9mth' ,
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
'onlinebank_credit_bill_min_amt_cur_minrepay_6mth'
]
feature_str = []

d = {}
c = {}
for i in feature_final:
    try:
        i_point = pd.read_excel('~/desktop/评分卡/复贷模型/result/%s_woe_point.xls' % i)
    except:
        logging.critical('%s_woe_point文件读取失败' % i)
    dwoe = {}
    for j in i_point.index:
        dwoe[i_point[i + '_cut'][j]] = i_point['point'][j]
    dwoe.setdefault(np.nan, min(dwoe.values()))
    cutpoint = i_point['cut_point'][0]
    reg = re.compile('[0-9]+[\.]?[0-9]*[e]?[-\+]?[0-9]*')
    cutoffpoints = list(map(float, re.findall(reg, cutpoint)))
    cutoffpoints.insert(0, -np.inf)
    cutoffpoints.append(np.inf)
    d[i] = dwoe
    c[i] = cutoffpoints


@app.route('/register', methods=['POST'])
def register():
    data = request.get_data().decode('utf-8')
    try:
        data = json.loads(data)
    except:
        logging.error('输入的json格式不对')
        return '输入的json格式不对'
    try:
        assert len(data) == 18, '输入变量长度不对'
    except Exception as e:
        logger.error(e)
        return '输入变量长度不对'
    data = pd.DataFrame(data, index=[0])
    data = data.applymap(lambda x: np.nan if x == 'null' else x)
    try:
        data = data[feature_final]
    except KeyError:
        logging.error('变量名输入错误')
        return '变量名输入错误'
    if sum(map(lambda x: type(x) != int and type(x) != float, data.values[0].tolist())) > 0:
        logger.error('输入变量值不对，空值写成null，否则为数字')
        return '输入变量值不对，空值写成null，否则为数字'
    for i in feature_final:
        try:
            data[i + '_cut'] = pd.cut(data[i], c[i])
            data[i + '_point'] = data[i + '_cut'].map(d[i])
        except KeyError:
            logging.critical('%s_woe_point文件读取失败' % i)
            return '%s_woe_point文件读取失败' % i
    feature_point = [i + '_point' for i in feature_final]
    data_point = data[feature_point].sum(axis=1)[0] + 678
    return str(data_point) + '\n'

if __name__ == '__main__':
    app.run()


