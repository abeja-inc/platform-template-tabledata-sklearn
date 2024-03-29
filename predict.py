import http
import os
import traceback
from io import BytesIO
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np


ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')
ABEJA_TRAINING_RESULT_DIR = os.getenv('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')
Path(ABEJA_TRAINING_RESULT_DIR).mkdir(exist_ok=True)

with open(os.path.join(ABEJA_TRAINING_RESULT_DIR, 'skf_env.json')) as f:
    skf_env = json.load(f)
    NFOLD = skf_env.get('NFOLD')
    IS_MULTI = "MULTI_CLASS" in skf_env or "SVC_MULTI_CLASS" in skf_env
    NUM_CLASS = skf_env.get('NUM_CLASS')
    cols_train = skf_env.get('cols_train')

models = []
for i in range(NFOLD):
    filename = os.path.join(ABEJA_TRAINING_RESULT_DIR, f'model_{i}.pkl')
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        models.append(model)


def handler(request, context):
    print(f'start prediction with request : {request}, context : {context}')
    if 'http_method' not in request:
        message = 'Error: Support only "abeja/all-cpu:19.04" or "abeja/all-gpu:19.04".'
        print(message)
        return {
            'status_code': http.HTTPStatus.BAD_REQUEST,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': message}
        }

    try:
        data = request.read()
        csvfile = BytesIO(data)
        
        X_test = pd.read_csv(csvfile, usecols=cols_train)[cols_train]
        
        if IS_MULTI:
            pred = np.zeros((len(X_test), NUM_CLASS))
            for model in models:
                pred += np.identity(NUM_CLASS)[model.predict(X_test)]
            pred = np.argmax(pred, axis=1)
        else:
            pred = np.zeros(len(X_test))
            for model in models:
                pred += model.predict(X_test)
            pred /= len(models)
        
        print(pred)
        
        X_test['pred'] = pred
        
        return {
            'status_code': http.HTTPStatus.OK,
            'content_type': 'application/json; charset=utf8',
            'content': {'result': X_test.values, 
                        'field': X_test.columns.tolist()}
        }
    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        return {
            'status_code': http.HTTPStatus.INTERNAL_SERVER_ERROR,
            'content_type': 'application/json; charset=utf8',
            'content': {'message': str(e)}
        }


