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
from datetime import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import logging
from logging.handlers import TimedRotatingFileHandler

import warnings
warnings.filterwarnings("ignore")

LOG_DIR = '/home/yedingda/flask/'
path = '/home/yedingda/flask'


def server_logger():
    logger = logging.getLogger('server')
    logger.setLevel("INFO")
    formatter = logging.Formatter('%(asctime)s - %(name)s %(process)d %(levelname)s  %(message)s')
    fh = TimedRotatingFileHandler(os.path.join(LOG_DIR, 'predict_service.log'), when='H', interval=1, backupCount=48)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

LOG = server_logger()
app = Flask(__name__)


def load_IndependModel(path,model_name,LOG):
    try:
        model = joblib.load(path + '/'+ model_name +'.pkl')
        #model = joblib.load(FILE_PATH + '/'+ model_name +'.m')
        return model
    except Exception as e:
        LOG.error('--error occurred when loading model pkl : ' + e)


#  必要参数校验
def id_isnull_check(df,errorMsg,LOG):
    result = []
    for col in df.columns:
        if df[col].isnull().values[0]:
            result.append(col)
    LOG.info('--input param check: columns with the null value ...')
    LOG.info(np.str(result))
    missing_necessary_cols = [col for col in ['idcard','realname','mobilePhone'] if col in result]
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
        LOG.info('--predicting name is  ' + np.str(df['realname'].values[0]))
        LOG.info('--predicting idcard is  ' + np.str(df['idcard'].values[0]))
        LOG.info('--predicting mobile is  ' + np.str(df['mobilePhone'].values[0]))
    except Exception as e:
        LOG.error('-- input param check : some  cols is missing.' )
        missing_necessary_cols = [col for col in ['idcard','realname','mobilePhone'] if col not in df.columns]
        errorMsg = 'necessary cols is missing : {0}'.format(np.str(missing_necessary_cols))
        returnCode = '1001'  # id缺失
        return errorMsg, returnCode
    else:
        try:
            df = df[feature_final]
            returnCode = '0'  # 预处理成功
            return errorMsg,returnCode
        except Exception as e:
            missing_feature_cols_list = [col for col in feature_final if col not in df.columns]
            errorMsg = 'feature cols is missing : {0}'.format(np.str(missing_feature_cols_list))
            returnCode = '1002'  # 模型参数缺失
            return errorMsg,returnCode
        # Log('-- input param check : feature cols is wrong...')
 #   else:
 #           data_isnull_check(df,LOG)
 #           dtypes_object_check(df,LOG)


def data_process(df,feature_final,errorMsg,LOG):
    df.replace('null', np.nan, inplace=True)
    df.replace('', np.nan, inplace=True)
    df = df.applymap(lambda x: np.nan if x is None else x)
    df[feature_final] = df[feature_final].applymap(float)
    try:
        errorMsg,returnCode = input_param_check(df,feature_final,errorMsg,LOG)
        LOG.info('--input param check finish.')
        return df, errorMsg, returnCode

    except Exception as e:
        LOG.error('--error occurred when data preprocessing : ' + e)
        errorMsg = 'data process error'
        returnCode = '1000'
        return df, errorMsg, returnCode


def mx_scores(y_predprob):
    """
    y_predprob : 预测成为坏用户的概率【p】
    odds = 好/坏 = （1-p）/p
    odds 翻倍 评分增加 50
    700 = A + Blog(10)
    750 = A + Blog(20)

    上面基础上增加或减少50
    """
    df = pd.DataFrame(y_predprob)
    scores = 533.91 + 72.13 * np.log(np.abs(y_predprob[0] / (1 - y_predprob[0])))
    # scores = 533.91 + 72.13 * np.log(np.abs(y_predprob / (1 - y_predprob)))
    if scores > 950:
        scores = 950
    elif scores < 350:
        scores = 350
    else:
        pass

    df['score'] = scores
    df.columns = ['prob', 'score']
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
        result['name'] = df['realname'].values[0]
        result['mobile'] = df['mobilePhone'].values[0]
        result['cash_loan_90d'] = df['cash_loan_90d'].values[0]
        result['one_city_max'] = df['one_city_max'].values[0]
        result['indirectcnt'] = df['indirectcnt'].values[0]
        result['taobao_suborder_sum_realtotal_success_150d'] = df['taobao_suborder_sum_realtotal_success_150d'].values[0]
        result['carrier_call_cnt_peer_city_all_90d'] = df['carrier_call_cnt_peer_city_all_90d'].values[0]
        result['carrier_call_sum_durationinsecond_telephone_all_30d'] = df['carrier_call_sum_durationinsecond_telephone_all_30d'].values[0]
        result['carrier_call_freq_days_work_all_150d'] = df['carrier_call_freq_days_work_all_150d'].values[0]
        result['carrier_call_cnt_peer_province_dialed_30d'] = df['carrier_call_cnt_peer_province_dialed_30d'].values[0]

        LOG.info('score caculate finished !')
        return result


def predict_flask(df,model_name,feature_final,path,load_IndependModel,LOG,mx_scores,errorMsg):
    try:
        LOG.info('--start preprocessing data ...')
        df,errorMsg,returnCode = data_process(df,feature_final,errorMsg,LOG)
        LOG.info('--preprocessing data finished.')

        LOG.info('--start loading model ...')
        model = load_IndependModel(path,model_name,LOG)

        LOG.info('--start calculate score ...')
        predict_result = model_preditc(df,model,feature_final,LOG,mx_scores)
        LOG.info('--calculate score finished.')

        return predict_result,errorMsg,returnCode

    except Exception as e:
        LOG.error('--model predict failed.')
        # errorMsg = 'model_predict_failed'
        predict_result = e
        returnCode = '1'
        return predict_result,errorMsg,returnCode


@app.route('/cqmodel', methods=['POST'])
def access_model_predict():
    start_time = datetime.now()
    data_dict = request.get_json(force=True)
    data_df = pd.DataFrame(data_dict, index=[0])
    try:
        feature_final = load_IndependModel(path,'feature',LOG)
        result,errorMsg,returnCode = predict_flask(data_df,'model',feature_final,path,load_IndependModel,LOG,mx_scores,errorMsg='')

        end_time = datetime.now()

        if returnCode != '0':
            times = '--predict failure used time  %d ms.' % ((end_time - start_time).seconds*1000 + (end_time - start_time).microseconds/1000)
            LOG.error(times)
        else:
            returnCode == '0'
            times = '--predict success used time  %d ms.' % ((end_time - start_time).seconds*1000 + (end_time - start_time).microseconds/1000)
            LOG.info(times)

    except:
        # resp = make_response('{"result":"NO"}')
        end_time = datetime.now()
        times = '--predict failure used time  %d ms.' % ((end_time - start_time).seconds*1000 + (end_time - start_time).microseconds/1000)
        LOG.error(times)

    finally:

        try:
            if len(result):
                resp = make_response('{"returnCode":"%s","id_card":"%s","name":"%s","mobile":"%s","score":"%d",'
                                     '"cash_loan_90d":"%.2f","one_city_max":"%.2f","indirectcnt":"%.2f",'
                                     '"taobao_suborder_sum_realtotal_success_150d":"%.2f",'
                                     '"carrier_call_cnt_peer_city_all_90d":"%.2f",'
                                     '"carrier_call_sum_durationinsecond_telephone_all_30d":"%.2f",'
                                     '"carrier_call_freq_days_work_all_150d":"%.2f",'
                                     '"carrier_call_cnt_peer_province_dialed_30d":"%.2f"}'
                    % (returnCode, result['idcard'].values[0], result['name'].values[0], result['mobile'].values[0],
                       result['score'].values[0], result['cash_loan_90d'].values[0], result['one_city_max'].values[0],
                       result['indirectcnt'].values[0],
                       result['taobao_suborder_sum_realtotal_success_150d'].values[0],
                       result['carrier_call_cnt_peer_city_all_90d'].values[0],
                       result['carrier_call_sum_durationinsecond_telephone_all_30d'].values[0],
                       result['carrier_call_freq_days_work_all_150d'].values[0],
                       result['carrier_call_cnt_peer_province_dialed_30d'].values[0],
                       ))

            else:
                # returnCode = '0'
                resp = make_response('{"returnCode":"%s","errorMsg":"%s" }' % (returnCode,errorMsg))
                LOG.error("empty code returned.")
        except Exception as e:
            returnCode = '1'
            errorMsg = 'no result'
            resp = make_response('{"returnCode":"%s","errorMsg":"%s" }' % (returnCode,errorMsg))
            LOG.error("result return error.")

    resp.headers['Content-Type'] = 'application/json'
    return resp


if __name__ == '__main__':
    HTTP_SERVER_HOST = "0.0.0.0"
    HTTP_SERVER_PORT = 6037
    app.run(host=HTTP_SERVER_HOST, port=HTTP_SERVER_PORT, debug=False)



