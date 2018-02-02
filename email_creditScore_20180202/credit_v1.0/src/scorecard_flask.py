# -*- coding : utf-8 -*-

from flask import Flask
from flask import request
from flask import make_response
import pandas as pd
import numpy as np
from os.path import abspath
from os.path import join
from os.path import dirname

import json
from scorecard_frame import *
from datetime import datetime
import log


LOG = log.server_logger()
app = Flask(__name__)

path = abspath(join(dirname(__file__),'..','..','..','email_creditScore'))
# path = '/zsd/'
# path = '/Users/wanghuanan/Desktop/zsd/'
# model_path =  path + '/credit_v1.0/models'


feature_final = ['mail_bill_min_ratio_nopay_6mth',
 'mail_bill_avg_nopayment_6mth',
 'email_billitem_freq_overdue_6mth',
 'payamt_6',
 'mail_card_sum_cashlimit_singlebank',
 'mail_bill_avg_lastpayment_6mth',
 'mail_bill_avg_minpayment_6mth',
 'email_billitem_max_single_freq_repay_6mth',
 'email_billitem_ratio_cnt_mthshopping_6mth',
 'mail_bill_min_lastpayment_6mth',
 'mail_card_min_limit_singlebank',
 'email_billitem_cnt_mthwithdraw_6mth',
 'mail_bill_min_minpayment_6mth',
 'mail_card_min_cashlimit_singlebank',
 'mail_bill_avg_ratio_repay_6mth',
 'email_billitem_max_single_amount_6mth',
 'max_overdue',
 'mail_bill_max_minpayment_6mth',
 'email_billitem_max_single_freq_overdue_6mth',
 'mail_bill_max_ratio_nopay_6mth',
 'payamt_5',
 'email_billitem_ratio_cnt_mthinstallment_6mth',
 'email_billitem_min_amount_shopping_6mth',
 'mail_card_sum_limit_singlebank',
 'mail_bill_sum_bill_6mth',
 'email_billitem_sum_amount_overdue_6mth',
 'mail_bill_min_nopayment_6mth']



feature_str = ['gender']



var_WOE_list = ['mail_bill_min_ratio_nopay_6mth_woe',
 'mail_bill_avg_nopayment_6mth_woe',
 'email_billitem_freq_overdue_6mth_woe',
 'payamt_6_woe',
 'mail_card_sum_cashlimit_singlebank_woe',
 'mail_bill_avg_lastpayment_6mth_woe',
 'mail_bill_avg_minpayment_6mth_woe',
 'email_billitem_max_single_freq_repay_6mth_woe',
 'email_billitem_ratio_cnt_mthshopping_6mth_woe',
 'mail_bill_min_lastpayment_6mth_woe',
 'mail_card_min_limit_singlebank_woe',
 'email_billitem_cnt_mthwithdraw_6mth_woe',
 'mail_bill_min_minpayment_6mth_woe',
 'mail_card_min_cashlimit_singlebank_woe',
 'mail_bill_avg_ratio_repay_6mth_woe',
 'email_billitem_max_single_amount_6mth_woe',
 'max_overdue_woe',
 'mail_bill_max_minpayment_6mth_woe',
 'email_billitem_max_single_freq_overdue_6mth_woe',
 'mail_bill_max_ratio_nopay_6mth_woe',
 'payamt_5_woe',
 'email_billitem_ratio_cnt_mthinstallment_6mth_woe',
 'email_billitem_min_amount_shopping_6mth_woe',
 'mail_card_sum_limit_singlebank_woe',
 'mail_bill_sum_bill_6mth_woe',
 'email_billitem_sum_amount_overdue_6mth_woe',
 'mail_bill_min_nopayment_6mth_woe']





def predict_flask(df,model_name1,model_name2,feature_final,path,load_IndependModel,LOG,var_WOE_list):
    try:
        LOG.info('--start preprocessing data ...')
        df = data_process(df,LOG)
        LOG.info('--preprocessing data finished.')
    except Exception,e:
        # print 'error occurred : %s' % e
        LOG.error('--preprocessing data failed.')

    else:
        LOG.info('--start loading model and calculate score...')
        predict_result = var_bins_map_woe_pkl(df,path,model_name1,model_name2,feature_final,LOG,var_WOE_list)
        LOG.info('--calculate score finished.')
        # user_id = df['bill_id'].values[0]

        return predict_result



@app.route('/api/email/email_creditScore/v1.0',methods=['POST'])
def access_model_predict():
    start_time = datetime.now()
    data_dict = request.get_json(force=True)
    data_df = pd.DataFrame(data_dict,index=[0])
    try:
        result = predict_flask(data_df,'careful_woe_dict_train_py27','lr_2018-02-02',feature_final,path,load_IndependModel,LOG,var_WOE_list)
        end_time = datetime.now()
        times = '--predict success used time  %d ms.' % ((end_time - start_time).seconds*1000 + (end_time - start_time).microseconds/1000)
        LOG.info(times)

    except:
        resp = make_response('{"result":"NO"}')
        end_time = datetime.now()
        times = '--predict failure used time  %d ms.' % ((end_time - start_time).seconds*1000 + (end_time - start_time).microseconds/1000)
        LOG.error(times)

    else:

        try:
            if len(result):
                resp = make_response('{"result":"OK","emailid":"%s","nameoncard":"%s","score":%d,"grade":"%s","limit":%d}'
                    % (result['emailid'].values[0],result['nameoncard'].values[0],result['score'],result['grade'].values[0],result['limit']))

            else:
                resp = make_response('{"result":"NO"}')
                LOG.error("empty code returned.")
        except:

            LOG.error("result return error.")

    resp.headers['Content-Type'] = 'application/json'
    return resp


if __name__ == '__main__':
    HTTP_SERVER_HOST = "0.0.0.0"
    HTTP_SERVER_PORT = 6033
    app.run(host=HTTP_SERVER_HOST, port=HTTP_SERVER_PORT, debug=False)

