#!/bin/bash
ANACONDA_HOME=/root/anaconda2
# ROOT_DIR=`realpath $(dirname $0)`
ROOT_DIR=/email_creditScore/credit_v1.0
# cd $ROOT_DIR/src && CUDA_VISIBLE_DEVICES="" $ANACONDA_HOME/bin/gunicorn -w 1 -D -b 0.0.0.0:6023 -p $ROOT_DIR/log/predicate_service.pid zsd_credit_model_flask:app
cd /email_creditScore/credit_v1.0/src && CUDA_VISIBLE_DEVICES="" $ANACONDA_HOME/bin/gunicorn -w 2 -D -b 0.0.0.0:6033 -p $ROOT_DIR/log/predicate_service.pid scorecard_flask:app
