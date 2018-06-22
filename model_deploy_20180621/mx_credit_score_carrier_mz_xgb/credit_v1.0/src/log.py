# -*- coding: utf-8 -*-
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from os.path import abspath
from os.path import dirname
from os.path import join

LOG_DIR=abspath(join((dirname(__file__)), '..', 'log'))

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