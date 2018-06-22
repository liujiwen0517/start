# -*- coding:utf-8 -*-
# lr
# 2018-6
# baron


import os
from os.path import abspath
from os.path import join
from os.path import dirname
import sys
import time

import numpy as np
import pandas as pd
from sklearn.externals import joblib


import warnings
warnings.filterwarnings("ignore")


path = abspath(join(dirname(__file__),'..','..','..','mx_credit_score_carrier_mz_xgb'))




def load_IndependModel(path,model_name,LOG):
    try:
        FILE_PATH = path + '/credit_v1.0/models'
        model = joblib.load(FILE_PATH + '/'+ model_name +'.pkl')
        #model = joblib.load(FILE_PATH + '/'+ model_name +'.m')
        return model
    except Exception as e:
        LOG.error('--error occurred when loading model pkl : ' + e)




def dtypes_object_check(df,LOG):
    """ dealing the dtyps of data"""
    columns = df.select_dtypes(include=['O']).columns
    result = []
    for col in columns:
        if col not in ['name','carrier_base_level']:
            result.append(col)

        else:
            pass

    LOG.info('--input param check: there are string types but not name and carrier_base_level ...')
    LOG.info(np.str(result))



# 必要参数校验
def id_isnull_check(df,errorMsg,LOG):
    result = []
    for col in df.columns:
        if df[col].isnull().values[0]:
            result.append(col)

    LOG.info('--input param check: columns with the null value ...')
    LOG.info(np.str(result))
    missing_necessary_cols = [col for col in ['idcard','name','mobile'] if col in result]
    if 'name' in result:
        LOG.error('-- input param check : name is null.')
    elif 'idcard' in result:
        LOG.error('-- input param check : idcard is null.')

    elif 'mobile' in result:
        LOG.error('-- input param check : mobile is null.')

    if len(result) > 0:
        errorMsg = np.str(missing_necessary_cols)
    return errorMsg


# 参数检查
def input_param_check(df,feature_final,errorMsg,LOG):
    try:
        errorMsg = id_isnull_check(df,errorMsg,LOG)
        LOG.info('--predicting name is  ' + np.str(df['name'].values[0]))
        LOG.info('--predicting idcard is  ' + np.str(df['idcard'].values[0]))
        LOG.info('--predicting mobile is  ' + np.str(df['mobile'].values[0]))
    except Exception as e:
        LOG.error('-- input param check : some  cols is missing.')
        missing_necessary_cols = [col for col in ['idcard','name','mobile'] if col not in df.columns]
        errorMsg = 'necessary cols is missing : {0}'.format(np.str(missing_necessary_cols))
        returnCode = '1001' # id缺失
        return errorMsg,returnCode

    else:
        try:
            df = df[feature_final]
            returnCode = '0'  # 预处理成功
            return errorMsg,returnCode
        except Exception as e:
            missing_feature_cols_list = [col for col in feature_final if col not in df.columns]
            errorMsg = 'feature cols is missing : {0}'.format(np.str(missing_feature_cols_list))
            returnCode = '1002' # 模型参数缺失
            return errorMsg,returnCode
        # Log('-- input param check : feature cols is wrong...')

 #   else:
 #           data_isnull_check(df,LOG)
 #           dtypes_object_check(df,LOG)



def carrier_base_level_mapping(df):
    dict_base = {'一星' : 1,'二星' :1,'三星': 1,'普通': 1,'银卡':1,
                 '四星':2,'五星':2,'六星':2,'金卡':2,'贵宾':2,'七星':3,
                 '未知':0}

    df['carrier_base_level'] = df['carrier_base_level'].map(dict_base)
    return df




def string_vars_deal(df,string_cols):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    for i in string_cols:
        df[i] = le.fit_transform(df[i])
        
    return df






def var_bins_map_woe_pkl(data,woe_dict,feature_selected):
    for col in feature_selected:
        print('the var for bins :{0}'.format(col))
        cutOffPoint = woe_dict[col]['cut_point'][0]
        print(cutOffPoint)
        woe_dict_new = {}
        woe_dict[col][col + '_cut'] = woe_dict[col][col + '_cut'].astype(np.str)
        tmp_dict = woe_dict[col][[col + '_cut','WOE']].to_dict(orient='index')
        for key in tmp_dict.keys():
            woe_dict_new[tmp_dict[key][col + '_cut']] = tmp_dict[key]['WOE']
            
        woe_dict_new_df = pd.DataFrame(woe_dict_new,index=[len(woe_dict_new)]).T
        if data[col].dtype == 'O':
            data[col + '_cut'] = data[col]
            data[col + '_cut'].fillna('NaN',inplace=True)
            data[col + '_cut'] = data[col + '_cut'].astype(np.str)
            woe_dict_new_ = woe_dict_new.copy()
            if 'nan' in woe_dict_new_.keys():
                woe_dict_new_['NaN'] = woe_dict_new_['nan']
            
        else:
            
            data[col + '_cut'],_ = pd.cut(data[col],bins=cutOffPoint,retbins=True)
            data[col + '_cut'] = data[col + '_cut'].astype(np.str)
            
        if data[col].dtype == 'O':
            data[col + '_woe'] = data[col + '_cut'].map(lambda x : woe_dict_new_[x])
            data[col + '_cut'].replace('NaN',np.nan,inplace=True)
            
        else:
            data[col + '_woe'] = data[col + '_cut'].map(lambda x : woe_dict_new[x]).astype(np.float64)
        if data[col + '_woe'].isnull().sum() > 0:
            data.ix[data[col + '_woe'].isnull(),col + '_woe'] = woe_dict_new_df[woe_dict_new_df.index.isnull()].values[0][0]
            
    return data 




def data_process(df,feature_final,errorMsg,LOG):
    try:
        errorMsg,returnCode = input_param_check(df,feature_final,errorMsg,LOG)
        LOG.info('--input param check finish.')
        # df = carrier_base_level_mapping(df)
        df = string_vars_deal(df,['province'])
        df.fillna(-99999999,inplace=True)
        return df,errorMsg,returnCode

    except Exception as e:
        LOG.error('--error occurred when data preprocessing : ' + e)
        errorMsg = 'carrier_base_level_mapping'
        return df,errorMsg,returnCode
    




def score_to_grade(score):

    """

tag 0   1   bad_rate
score_cut           
(350, 500]  965 830 0.462396
(500, 550]  490 251 0.338731
(550, 600]  440 181 0.291465
(600, 650]  331 102 0.235566
(650, 700]  231 49  0.175000
(700, 950]  226 32  0.124031

score_line ≥ 650   pass_rate 11%   bad_rate 15%

-

    """
    if (score >= 350) & (score < 551):
        return "Very_Bad"
    elif (score >= 551) & (score <601):
        return "Poor"
    elif (score >= 601) & (score <651):
        return "Fair"
    elif (score >= 651) & (score <701):
        return "Good"
    elif (score >= 701) & (score <801):
        return "Very_Good"
    elif (score >= 801) & (score <=950):
        return "Excellent"
    else:
        return "unknow"




def mx_scores(y_predprob):
    """
    y_predprob : 预测成为坏用户的概率【p】
    odds = 好/坏 = （1-p）/p

    odds 翻倍 评分增加 50

    700 = A + Blog(10)
    750 = A + Blog(20)
    
    上面基础上增加或减少50
    """
    df = pd.DataFrame(y_predprob,columns=['predprob'])
    y_predprob = df['predprob'].values
    scores = 391.06 + 134.17*np.log(np.abs((1- y_predprob))/y_predprob)
    for index in range(scores.shape[0]):
        if (scores[index] == np.inf) | (scores[index] == -np.inf):
            scores[index] = 0
        else:
            pass
    scores = np.array([round(sco) for sco in scores])
    scores = list(scores.astype(np.int))
    for index in range(len(scores)):
        if (scores[index] < 350) and scores[index] > 0:
            scores[index] = 350

        elif scores[index] > 950:
            scores[index] = 950

    df['score'] = scores
    df['grade'] = df['score'].apply(score_to_grade)
    df['decision'] = df['score'].apply(lambda x : 'REJECT' if x < 670 else 'APPROVE')


    
    return df






def model_preditc(df,model,feature_final,LOG,mx_scores):
    try:
        y_predprob = model.predict_proba(df[feature_final])[:, 1]

    except Exception as e:
        LOG.error('--error occurred when model predict : ' + e)

    else:
        LOG.info('model predict finished !')
        result = mx_scores(y_predprob)
        result['idcard'] = df['idcard'].values[0]
        result['name'] = df['name'].values[0]
        result['mobile'] = df['mobile'].values[0]
        LOG.info('score caculate finished !')
        return result






