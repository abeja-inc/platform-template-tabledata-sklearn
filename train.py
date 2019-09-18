# coding: utf-8
# Template: 

import os
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorboardX import SummaryWriter

from callbacks import Statistics
from data_loader import train_data_loader
from parameters import Parameters


ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')
Path(ABEJA_TRAINING_RESULT_DIR).mkdir(exist_ok=True)

DATALAKE_CHANNEL_ID = Parameters.DATALAKE_CHANNEL_ID
DATALAKE_TRAIN_FILE_ID = Parameters.DATALAKE_TRAIN_FILE_ID
DATALAKE_VAL_FILE_ID = Parameters.DATALAKE_VAL_FILE_ID
INPUT_FIELDS = Parameters.INPUT_FIELDS
LABEL_FIELD = Parameters.LABEL_FIELD
PARAMS = Parameters.as_params()
IS_MULTI = "multi_class" in PARAMS
NUM_CLASS = Parameters.NUM_CLASS

statistics = Statistics(Parameters.NFOLD)

log_path = os.path.join(ABEJA_TRAINING_RESULT_DIR, 'logs')
writer = SummaryWriter(log_dir=log_path)

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

if IS_MULTI:
    evaluator = accuracy_score
else:
    evaluator = roc_auc_score


def handler(context):
    print(f'start training with parameters : {Parameters.as_dict()}, context : {context}')
    
    X_train, y_train, cols_train = train_data_loader(
        DATALAKE_CHANNEL_ID, DATALAKE_TRAIN_FILE_ID, LABEL_FIELD, INPUT_FIELDS)
    models = []
    pred = np.zeros(len(X_train))

    if DATALAKE_VAL_FILE_ID:
        X_val, y_val, _ = train_data_loader(
            DATALAKE_CHANNEL_ID, DATALAKE_VAL_FILE_ID, LABEL_FIELD, INPUT_FIELDS)
        if IS_MULTI:
            pred_val = np.zeros((len(X_val), NUM_CLASS))
        else:
            pred_val = np.zeros(len(X_val))
    else:
        X_val, y_val, pred_val = None, None, None

    for i, (train_index, valid_index) in enumerate(skf.split(X_train, y_train)):
        model = classifier(**PARAMS)
        model.fit(X_train.iloc[train_index], y_train[train_index])
        pred[valid_index] = model.predict(X_train.iloc[valid_index])

        score = evaluator(y_train[valid_index], pred[valid_index])
        score_val = 0.0

        filename = os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.pkl')
        pickle.dump(model, open(filename, 'wb'))
        
        models.append(model)

        if DATALAKE_VAL_FILE_ID:
            pred_val_cv = model.predict(X_val)
            if IS_MULTI:
                pred_val += np.identity(NUM_CLASS)[pred_val_cv]
            else:
                pred_val += pred_val_cv
            score_val = evaluator(y_val, pred_val_cv)

        print('-------------')
        print('cv {} || score:{:.4f} || val_score:{:.4f}'.format(i + 1, score, score_val))
        writer.add_scalar('main/acc', score, i + 1)
        writer.add_scalar('test/acc', score_val, i + 1)
        statistics(i + 1, None, score, None, score_val)

    score = evaluator(y_train, pred)
    score_val = 0.0

    if DATALAKE_VAL_FILE_ID:
        if IS_MULTI:
            pred_val = np.argmax(pred_val, axis=1)
        else:
            pred_val /= len(models)
        score_val = evaluator(y_val, pred_val)

    print('-------------')
    print('cv total score:{:.4f} || cv total val_score:{:.4f}'.format(score, score_val))
    statistics(Parameters.NFOLD, None, score, None, score_val)
    writer.add_scalar('main/acc', score, Parameters.NFOLD)
    writer.add_scalar('test/acc', score_val, Parameters.NFOLD)

    di = {
        **(Parameters.as_dict()),
        'cols_train': cols_train
    }
    skf_env = open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'skf_env.json'), 'w')
    json.dump(di, skf_env)
    skf_env.close()
    return


if __name__ == '__main__':
    handler(None)
