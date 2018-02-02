# -*- coding : utf-8 -*-
import os
from os.path import abspath
from os.path import join
from os.path import dirname
import sys
import time

import re
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")


path = abspath(join(dirname(__file__),'..','..','..','email_creditScore'))


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





def load_IndependModel(path,model_name,LOG):
    try:
        FILE_PATH = path + '/credit_v1.0/models'
        model = joblib.load(FILE_PATH + '/'+ model_name +'.pkl')
        #model = joblib.load(FILE_PATH + '/'+ model_name +'.m')
        return model
    except Exception,e:
        LOG.error('--error occurred when loading model pkl : ' + e)


def dtypes_object_check(df,LOG):
    """ dealing the dtyps of data"""
    columns = df.select_dtypes(include=['O']).columns
    result = []
    for col in columns:
        if col not in ['emailid','nameoncard']:
            result.append(col)

        else:
            pass

    LOG.info('--input param check: there are string types but not emailid and nameoncard ...')
    LOG.info(np.str(result))


def data_isnull_check(df,LOG):
    result = []
    for col in df.columns:
        if df[col].isnull().values[0]:
            result.append(col)

    LOG.info('--input param check: columns with the null value ...')
    LOG.info(np.str(result))
    if 'gender' in result:
        LOG.error('-- input param check : gender is null.')
    elif 'appl_time_hour' in result:
        LOG.error('-- input param check : appl_time_hour is null.')



def input_param_check(df,LOG):
    try:
        LOG.info('--predicting emailid is  ' + np.str(df['emailid'].values[0]))
        LOG.info('--predicting nameoncard is  ' + np.str(df['nameoncard'].values[0]))
        cols = ['mail_bill_min_ratio_nopay_6mth',
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
        df = df[cols]
    except Exception,e:
        LOG.error('-- input param check : some feature cols is missing.')
        # Log('-- input param check : feature cols is wrong...')

    else:
            data_isnull_check(df,LOG)
            dtypes_object_check(df,LOG)




def data_process(df,LOG):
    try:
        input_param_check(df,LOG)
        LOG.info('--input param check finish.')
        return df

    except Exception,e:
        LOG.error('--error occurred when data preprocessing : ' + e)
        # print 'error occurred when data preprocessing : %s' % e


#
# def var_bins_map_woe_pkl(data,woe_dict,feature_selected):
#     for col in feature_selected:
#         print(col)
#         cutOffPoints = woe_dict[col]['cut_point'][0]
#         # print(cutOffPoints)
#         woe_dict_new = {}
#         woe_dict[col][col + '_cut'] = woe_dict[col][col + '_cut'].astype(np.str)
#         tmp_dict = woe_dict[col][[col + '_cut','WOE']].to_dict(orient='index')
#         for key in tmp_dict.keys():
#             woe_dict_new[tmp_dict[key][col + '_cut']] = tmp_dict[key]['WOE']
#
#         # print(woe_dict_new)
#         woe_dict_new_df = pd.DataFrame(woe_dict_new,index=[len(woe_dict_new)]).T
#         data[col + '_cut'],_ = pd.cut(data[col], bins=cutOffPoints, retbins=True,right=False)
#         data[col + '_cut'] = data[col + '_cut'].astype(np.str)
#         # data[col + '_cut'].replace('nan',np.nan,inplace=True)
#         # print(data[col + '_cut'].value_counts())
#         data[col + '_woe'] = data[col + '_cut'].map(lambda x: woe_dict_new[x]).astype(np.float64)
#         if data[col + '_woe'].isnull().sum() > 0:
#             data.ix[data[col + '_woe'].isnull(),col + '_woe'] = woe_dict_new_df[woe_dict_new_df.index.isnull()].values[0][0]


def var_bins_map_woe_pkl(data,path,model_name1,model_name2,feature_selected,LOG,var_WOE_list):
    try:
        woe_dict = load_IndependModel(path,model_name1,LOG)
        for col in feature_selected:
            cutOffPoints = woe_dict[col]['cut_point'][0]
            woe_dict_new = {}
            woe_dict[col][col + '_cut'] = woe_dict[col][col + '_cut'].astype(np.str)
            tmp_dict = woe_dict[col][[col + '_cut','WOE']].to_dict(orient='index')
            for key in tmp_dict.keys():
                woe_dict_new[tmp_dict[key][col + '_cut']] = tmp_dict[key]['WOE']
            woe_dict_new_df = pd.DataFrame(woe_dict_new,index=[len(woe_dict_new)]).T
            data[col + '_cut'],_ = pd.cut(data[col], bins=cutOffPoints, retbins=True,right=False)
            data[col + '_cut'] = data[col + '_cut'].astype(np.str)
            data[col + '_woe'] = data[col + '_cut'].map(lambda x: woe_dict_new[x]).astype(np.float64)
            if data[col + '_woe'].isnull().sum() > 0:
                data.ix[data[col + '_woe'].isnull(),col + '_woe'] = woe_dict_new_df[woe_dict_new_df.index.isnull()].values[0][0]

        LR_model_2_fit = load_IndependModel(path,model_name2,LOG)
        y_pred = LR_model_2_fit.predict_proba(data[var_WOE_list])[:, 1]
        result = mx_scores_v2(y_pred)
        result['emailid'] = data['emailid'].values[0]
        result['nameoncard'] = data['nameoncard'].values[0]
        return result
    except Exception,e:
        LOG.error('--error occurred when load model : ' + e)







def score_to_grade_v2(score):

    """
# label           0      1  interval_bad_rate
# bins
# [1, 51)        84  18132               1.00
# [51, 101)      57   4467               0.99
# [101, 151)     23    640               0.96
# [151, 201)     66   3719               0.98
# [201, 251)    312   8580               0.96
# [251, 301)    518   8243               0.94
# [301, 351)    807   2527               0.76
# [351, 401)   1359   1780               0.57
# [401, 451)   5659   9122               0.62
# Very_Bad

# [451, 501)   9434   3181               0.25
# [501, 551)  15006   3116               0.17
# Poor

# [551, 601)  20854   2272               0.10
# Fair

# [601, 651)  51160   2419               0.05
# Good

# [651, 701)  48817   1736               0.03
# [701, 751)  55077   1017               0.02
# Very_Good

# [751, 801)  24421    172               0.01
# [801, 851)   2392      9               0.00
# [851, 1000)   2392      9               0.00
# Excellent

    """
    if (score >= 1) & (score < 451):
        return "Very_Bad"
    elif (score >= 451) & (score <551):
        return "Poor"
    elif (score >= 551) & (score <601):
        return "Fair"
    elif (score >= 601) & (score <651):
        return "Good"
    elif (score >= 651) & (score <751):
        return "Very_Good"
    elif (score >= 751) & (score <=1000):
        return "Excellent"
    else:
        return "unknow"

def grade_to_limit_v2(grade):
    if grade == "Good":
        return 3000
    elif grade == 'Very_Good':
        return 10000
    elif grade == 'Excellent':
        return 30000
    else:
        return 0



def mx_scores_v2(y_predprob):
    """
    score : 350 - 1000
    700 = A - B log(s)
    700 - 50 = A - B log(2s)

    """
    df = pd.DataFrame(y_predprob,columns=['predprob'])
    y_predprob = df['predprob'].values
    scores = 541.51 - 72.13*np.log(y_predprob/np.abs((1- y_predprob)))
    for index in range(scores.shape[0]):
        if (scores[index] == np.inf) | (scores[index] == -np.inf):
            scores[index] = 1
        else:
            pass
    scores = np.array([round(sco) for sco in scores])
    scores = list(scores.astype(np.int))
    for index in range(len(scores)):
        if scores[index] < 1:
            scores[index] = 1

        elif scores[index] > 1000:
            scores[index] = 1000

    df['score'] = scores
    # df.rename(columns= {'tag' : 'label'},inplace=True)
    # result = np.c_[y_predprob,scores]
    # result = pd.DataFrame(result,columns=['predict_proba','score','label'])
    df['grade'] = df['score'].apply(score_to_grade_v2)
    df['limit'] = df['grade'].apply(grade_to_limit_v2)
    return df



















