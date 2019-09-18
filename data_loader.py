import os
from pathlib import Path

import pandas as pd
from abeja.datalake import Client as DatalakeClient


ABEJA_STORAGE_DIR_PATH = os.getenv('ABEJA_STORAGE_DIR_PATH', '~/.abeja/.cache')


def train_data_loader(channel_id: str, file_id: str, label_field: str, input_fileds: list = None):
    datalake_client = DatalakeClient()
    channel = datalake_client.get_channel(channel_id)
    datalake_file = channel.get_file(file_id)
    datalake_file.get_content(cache=True)

    csvfile = Path(ABEJA_STORAGE_DIR_PATH, channel_id, file_id)
    if input_fileds:
        train = pd.read_csv(csvfile, usecols=input_fileds+[label_field])
    else:
        train = pd.read_csv(csvfile)

    y_train = train[label_field].values
    cols_drop = [c for c in train.columns if train[c].dtype == 'O'] + [label_field]
    train.drop(cols_drop, axis=1, inplace=True)
    X_train = train
    cols_train = X_train.columns.tolist()
    del train
    return X_train, y_train, cols_train
