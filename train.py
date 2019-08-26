# coding: utf-8
# Template: 

import os
import gc
import json
from pathlib import Path
from math import modf
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import StratifiedKFold

from abeja.datalake import Client as DatalakeClient



# Configurable, but you don't need to set this. Default is "~/.abeja/.cache" in ABEJA Platform.
ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR')

DATALAKE_CHANNEL_ID = os.getenv('DATALAKE_CHANNEL_ID')
DATALAKE_TRAIN_ITEM_ID = os.getenv('DATALAKE_TRAIN_ITEM_ID')
DATALAKE_TEST_ITEM_ID = os.getenv('DATALAKE_TEST_ITEM_ID')

# params for load
INPUT_FIELDS = os.getenv('INPUT_FIELDS')
TARGET_FIELD = os.getenv('TARGET_FIELD') # required

if TARGET_FIELD is None:
    raise Exception(f'TARGET is required')

# params for clf
PARAMS = os.getenv('PARAMS')
params = {}
if PARAMS is None:
    pass
elif len(PARAMS) == 0:
    pass
else:
    for kv in PARAMS.split(','):
        k, v = kv.split('=')
        
        try:
            if v in ['True', 'False']:
                params[k] = bool(v)
            elif v == 'None':
                params[k] = None
            else:
                # int or float
                decimal, integer = modf(float(v))
                if decimal == 0:
                    params[k] = int(v)
                else:
                    params[k] = float(v)
        except:
            params[k] = v

NFOLD = int(os.getenv('NFOLD', '5'))
skf = StratifiedKFold(n_splits=NFOLD)

CLF = os.getenv('CLF')
if CLF is None:
    model = LinearRegression
    
elif CLF == 'LinearRegression':
    classifier = LinearRegression
    
elif CLF == 'LogisticRegression':
    classifier = LogisticRegression
    
elif CLF == 'SVR':
    classifier = SVR

elif CLF == 'SVC':
    classifier = SVC


def handler(context):
    print('Start train handler.')
    
    # load train
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
    datalake_file = channel.get_file(DATALAKE_TRAIN_ITEM_ID)
    datalake_file.get_content(cache=True)
    
    csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_ITEM_ID)
    if INPUT_FIELDS is None:
        train = pd.read_csv(csvfile)
    else:
        usecols = INPUT_FIELDS.split(',')
        train = pd.read_csv(csvfile, usecols=usecols+[TARGET_FIELD])
    
    y_train = train[TARGET_FIELD].values
    cols_drop = [c for c in train.columns if train[c].dtype == 'O'] + [TARGET_FIELD]
    train.drop(cols_drop, axis=1, inplace=True)
    X_train = train
    cols_train = X_train.columns.tolist()
    del train
    
    models = []
    pred = np.zeros(len(X_train))
    for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        model = classifier(**params)
        model.fit(X_train.iloc[train_index], y_train[train_index])
        pred[valid_index] = model.predict(X_train.iloc[valid_index])
        
        filename = os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.pkl')
        pickle.dump(model, open(filename, 'wb'))
        
        models.append(model)
    
    di = {
            'NFOLD': NFOLD,
            'CLF': CLF,
            'cols_train': cols_train
            
        }
    skf_env = open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'skf_env.json'), 'w')
    json.dump(di, skf_env)
    skf_env.close()
    
    del X_train; gc.collect()
    
    # load test
    if DATALAKE_TEST_ITEM_ID is not None:
        datalake_client = DatalakeClient()
        channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
        datalake_file = channel.get_file(DATALAKE_TEST_ITEM_ID)
        datalake_file.get_content(cache=True)
        
        csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TEST_ITEM_ID)
        X_test = pd.read_csv(csvfile, usecols=cols_train)[cols_train]
        
        pred = np.zeros(len(X_test))
        for model in models:
            pred += model.predict(X_test)
        pred /= len(models)
        
        print(pred)
    
    
    return

