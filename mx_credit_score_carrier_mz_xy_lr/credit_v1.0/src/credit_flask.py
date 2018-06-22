# -*- coding : utf-8 -*-

# xgb

from flask import Flask
from flask import request
from flask import make_response
import pandas as pd
import numpy as np
from os.path import abspath
from os.path import join
from os.path import dirname

import json
from frame import *
from datetime import datetime
import log


LOG = log.server_logger()
app = Flask(__name__)

path = abspath(join(dirname(__file__),'..','..','..','mx_credit_score_carrier_mz_xy_lr'))






def predict_flask(df,model_name,feature_final,path,load_IndependModel,LOG,mx_scores,var_bins_map_woe_pkl,woe_dict_name,errorMsg):
    try:
        LOG.info('--start preprocessing data ...')
        df,errorMsg,returnCode = data_process(df,feature_final,errorMsg,LOG)
        LOG.info('--preprocessing data finished.')

        LOG.info('--start loading model ...')
        model = load_IndependModel(path,model_name,LOG)

        LOG.info('-- start loading vars bins mapping ...')
        woe_dict = load_IndependModel(path,woe_dict_name,LOG)

        LOG.info('-- start bins with careful bins ...')
        df = var_bins_map_woe_pkl(df,woe_dict,feature_final)


        LOG.info('--start calculate score ...')
        predict_result = model_preditc(df,model,feature_final,LOG,mx_scores)
        LOG.info('--calculate score finished.')

        return predict_result,errorMsg,returnCode


    except Exception as e:
        LOG.error('--model predict failed.')
        # errorMsg = 'model_predict_failed'
        predict_result = ''
        returnCode = '1'
        return predict_result,errorMsg,returnCode









@app.route('/api/carrier_mz_xy/risk/score/v1.1',methods=['POST'])
def access_model_predict():
    start_time = datetime.now()
    data_dict = request.get_json(force=True)
    data_df = pd.DataFrame(data_dict,index=[0])
    try:
        feature_final = load_IndependModel(path,'var_into_model_new',LOG)
        result,errorMsg,returnCode = predict_flask(data_df,'LR_model_2_fit_new',feature_final,path,load_IndependModel,LOG,mx_scores,var_bins_map_woe_pkl,'woe_dict_careful_train_20180621',errorMsg = '')
        end_time = datetime.now()
        # print(returnCode)

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
                resp = make_response('{"returnCode":"%s","idcard":"%s","name":"%s","mobile":"%s","score":%d,"grade":"%s","decision":"%s"}'
                    % (returnCode,result['idcard'].values[0],result['name'].values[0],result['mobile'].values[0],result['score'],result['grade'].values[0],result['decision'].values[0]))

            else:
                # returnCode = '0'
                resp = make_response('{"returnCode":"%s","errorMsg":"%s" }' % (returnCode,errorMsg))
                LOG.error("empty code returned.")
        except:
            returnCode = '1'
            errorMsg = 'result_return_error'
            resp = make_response('{"returnCode":"%s","errorMsg":"%s" }' % (returnCode,errorMsg))
            LOG.error("result return error.")

    resp.headers['Content-Type'] = 'application/json'
    return resp


if __name__ == '__main__':
    HTTP_SERVER_HOST = "0.0.0.0"
    HTTP_SERVER_PORT = 6039
    app.run(host=HTTP_SERVER_HOST, port=HTTP_SERVER_PORT, debug=False)

