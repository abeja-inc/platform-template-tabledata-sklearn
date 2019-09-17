# coding: utf-8
# Template: 

import os
import gc
import json
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from abeja.datalake import Client as DatalakeClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import StratifiedKFold

from parameters import Parameters


ABEJA_STORAGE_DIR_PATH = Parameters.ABEJA_STORAGE_DIR_PATH
ABEJA_TRAINING_RESULT_DIR = Parameters.ABEJA_TRAINING_RESULT_DIR
Path(ABEJA_TRAINING_RESULT_DIR).mkdir(exist_ok=True)

DATALAKE_CHANNEL_ID = Parameters.DATALAKE_CHANNEL_ID
DATALAKE_TRAIN_FILE_ID = Parameters.DATALAKE_TRAIN_FILE_ID
DATALAKE_TEST_FILE_ID = Parameters.DATALAKE_TEST_FILE_ID
INPUT_FIELDS = Parameters.INPUT_FIELDS
LABEL_FIELD = Parameters.LABEL_FIELD
PARAMS = Parameters.as_params()

skf = StratifiedKFold(n_splits=Parameters.NFOLD)

if Parameters.CLASSIFIER is None:
    classifier = LinearRegression
elif Parameters.CLASSIFIER == 'LinearRegression':
    classifier = LinearRegression
elif Parameters.CLASSIFIER == 'LogisticRegression':
    classifier = LogisticRegression
elif Parameters.CLASSIFIER == 'SVR':
    classifier = SVR
elif Parameters.CLASSIFIER == 'SVC':
    classifier = SVC
elif Parameters.CLASSIFIER == 'LinearSVR':
    classifier = LinearSVR
elif Parameters.CLASSIFIER == 'LinearSVC':
    classifier = LinearSVC


def handler(context):
    print(f'start training with parameters : {Parameters.as_dict()}, context : {context}')
    
    # load train
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
    datalake_file = channel.get_file(DATALAKE_TRAIN_FILE_ID)
    datalake_file.get_content(cache=True)
    
    csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_FILE_ID)
    if INPUT_FIELDS:
        train = pd.read_csv(csvfile, usecols=INPUT_FIELDS+[LABEL_FIELD])
    else:
        train = pd.read_csv(csvfile)

    y_train = train[LABEL_FIELD].values
    cols_drop = [c for c in train.columns if train[c].dtype == 'O'] + [LABEL_FIELD]
    train.drop(cols_drop, axis=1, inplace=True)
    X_train = train
    cols_train = X_train.columns.tolist()
    del train
    
    models = []
    pred = np.zeros(len(X_train))
    for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        print('cv {}'.format(i + 1))
        model = classifier(**PARAMS)
        model.fit(X_train.iloc[train_index], y_train[train_index])
        pred[valid_index] = model.predict(X_train.iloc[valid_index])
        
        filename = os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.pkl')
        pickle.dump(model, open(filename, 'wb'))
        
        models.append(model)
    
    di = {
        **(Parameters.as_dict()),
        'cols_train': cols_train
    }
    skf_env = open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'skf_env.json'), 'w')
    json.dump(di, skf_env)
    skf_env.close()
    
    del X_train; gc.collect()
    
    # load test
    if DATALAKE_TEST_FILE_ID is not None:
        print("Run for test file.")
        datalake_client = DatalakeClient()
        channel = datalake_client.get_channel(DATALAKE_CHANNEL_ID)
        datalake_file = channel.get_file(DATALAKE_TEST_FILE_ID)
        datalake_file.get_content(cache=True)
        
        csvfile = Path(ABEJA_STORAGE_DIR_PATH, DATALAKE_CHANNEL_ID, DATALAKE_TEST_FILE_ID)
        X_test = pd.read_csv(csvfile, usecols=cols_train)[cols_train]
        
        pred = np.zeros(len(X_test))
        for model in models:
            pred += model.predict(X_test)
        pred /= len(models)
        
        print(pred)
    
    
    return


if __name__ == '__main__':
    handler(None)
